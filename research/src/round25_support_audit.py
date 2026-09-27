"""One fixed, target-free R25 address support inventory; never a price experiment.

--run is deliberately separate from --self-check. Reads three saved 55-column
parsed frames and frozen row indices only. Does not import a target loader,
recompute duplicate removal, fit a model, or score prices.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import platform
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round25_research'
JSON_PATH = OUT / 'address_support.json'
MD_PATH = ROOT / 'reports/round25_address_support.md'
FEATURE_SOURCE_SHA256 = 'deb3be7732bc5fd1be0a117afc9d762fc374e8144be35def35fa85bd41e511de'
SNAPSHOT_SHA256 = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
SPLITS = ('train', 'valid', 'test')
MODEL_STAGE = 'development'
QUANTILES = (0.0, .25, .5, .75, .9, .99, 1.0)
FEATURE_COLUMNS = ['addr_lane_path', 'addr_house_block100', 'addr_detail_eligible']
AUDIT_RULES = {
    'scope': 'canonical train and frozen development split; valid/test coverage only',
    'support_reference_pairs': ['development_fit_to_query', 'canonical_train_to_test'],
    'support_row_diagnostic': 10, 'block_distinct_main_number_diagnostic': 2,
    'main_number_quantiles': list(QUANTILES),
    'quantile_definition': 'empirical lower order statistic floor(q*(n-1)); Python integers, no clipping',
    'no_rule_or_bucket_selection_from_inventory': True,
    'bucket_width_is_fixed_by_feature_module': 100,
    'examples_maximum_total': 40, 'examples_per_population_reason': 2,
    'examples_maximum_per_population': 10,
    'reconstruction_sample_maximum_per_split': 128,
    'reconstruction_spaced_rows_per_split': 64,
    'sample_selection': 'fixed first two per reason/eligible-detail flags, then evenly spaced rows',
    'support_excludes_ineligible_rows': True,
    'row_support_is_not_distinct_project_or_physical_distance': True,
}


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


class Bindings:
    def __init__(self):
        self.files = {}

    def bind(self, name, expected=None):
        path = (ROOT / name).resolve()
        need(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Unsafe/missing input: ' + str(name))
        key = rel(path)
        if key not in self.files:
            self.files[key] = {'path': key, 'sha256': sha(path), 'bytes': path.stat().st_size}
        item = self.files[key]
        need(expected is None or item['sha256'] == expected, 'Changed input: ' + key)
        return item

    def json(self, name, expected=None, receipt=False):
        item = self.bind(name, expected)
        path = ROOT / item['path']
        if receipt:
            receipt_item = self.bind(rel(path.with_suffix('.sha256')))
            need((ROOT / receipt_item['path']).read_text(encoding='ascii').strip() == item['sha256'],
                 'Receipt differs: ' + item['path'])
        return json.loads(path.read_text(encoding='utf-8-sig'))

    def verify_unchanged(self):
        after = {}
        for name, before in self.files.items():
            path = ROOT / name
            after[name] = {'path': name, 'sha256': sha(path), 'bytes': path.stat().st_size}
            need(after[name] == before, 'Input changed during inventory: ' + name)
        return after


def snapshot_paths(snapshot):
    known = {name: item['original_sha256'] for name, item in snapshot['copied_files'].items()}
    for item in snapshot['external_dependencies']:
        name, digest = item['original_path'], item['sha256']
        need(name not in known or known[name] == digest, 'Snapshot conflict: ' + name)
        known[name] = digest
    return known


def check_index(values, size, count, label):
    need(isinstance(values, np.ndarray) and values.dtype == np.dtype('int64')
         and values.shape == (count,), 'Index dtype/shape differs: ' + label)
    need(((values >= 0) & (values < size)).all() and len(np.unique(values)) == count,
         'Index bounds/uniqueness differ: ' + label)
    need(count < 2 or np.all(values[1:] > values[:-1]), 'Index must preserve increasing source order: ' + label)


def canonical_positions(cache_indices, keep, fi, qi):
    need(cache_indices.dtype == np.dtype('int64')
         and np.array_equal(cache_indices, np.arange(len(cache_indices), dtype='int64')),
         'Cleaned-cache index is not a complete one-to-one sequence')
    check_index(keep, len(cache_indices), len(keep), 'canonical keep')
    check_index(fi, len(keep), len(fi), 'fit')
    check_index(qi, len(keep), len(qi), 'query')
    need(not np.intersect1d(fi, qi).size, 'Fit and query overlap')
    return keep[fi], keep[qi]


def check_parsed(frame, entry, parser):
    need(isinstance(frame, pd.DataFrame) and list(frame.columns) == entry['parsed_columns']
         and len(frame.columns) == 55, 'Parsed column schema differs')
    need({name: str(dtype) for name, dtype in frame.dtypes.items()} == entry['parsed_dtypes'],
         'Parsed dtype schema differs')
    need(not {'總價元', 'y', 'target'} & set(frame.columns), 'Target column in geographic frame')
    n = entry['original_rows']
    need(len(frame) == n == entry['rows_preserved_in_output'] and frame.index.equals(pd.RangeIndex(n)),
         'Original parsed row order differs')
    positions = np.arange(n, dtype='int64')
    need(np.array_equal(frame.source_row_index.to_numpy(), positions)
         and np.array_equal(frame.source_excel_row.to_numpy(), positions + 2), 'Original row provenance differs')
    header = frame.is_embedded_header.to_numpy(dtype=bool)
    need(int(header.sum()) == entry['embedded_header_rows']
         and frame.loc[header, 'source_excel_row'].tolist() == entry['embedded_header_excel_rows'],
         'Embedded-header evidence differs')
    need(frame.loc[header, 'existing_feature_cache_row_index'].eq(-1).all(), 'Header must have cache index -1')
    clean = frame.loc[~header].sort_values('existing_feature_cache_row_index').reset_index(drop=True)
    need(len(clean) == entry['existing_feature_cache_rows']
         and np.array_equal(clean.existing_feature_cache_row_index.to_numpy(), np.arange(len(clean), dtype='int64')),
         'Cache mapping is incomplete or reordered')
    need(frame.source_id.isna().sum() == entry['source_id_missing']
         and frame.source_id.duplicated().sum() == entry['duplicate_ids_preserved_excess'],
         'Original ID provenance differs; IDs are not join keys')
    need(frame.source_address.map(parser.normalize_text).equals(frame.address_normalized)
         and frame.source_district.map(parser.normalize_admin_name).equals(frame.supplied_district_normalized),
         'Stored normalized address/district differs from original X provenance')
    for col in ('admin_query_eligible', 'road_query_eligible', 'house_query_eligible'):
        need(not frame.loc[header, col].any(), 'Header unexpectedly eligible')
    return clean


def check_features(frame, features, details):
    need(list(features.columns) == FEATURE_COLUMNS and features.index.equals(frame.index)
         and details.index.equals(frame.index), 'Feature/details alignment differs')
    need(str(features.addr_detail_eligible.dtype) == 'float32'
         and features.addr_detail_eligible.isin([0.0, 1.0]).all(), 'Eligibility feature dtype/value differs')
    for col in FEATURE_COLUMNS[:2]:
        need(str(features[col].dtype) == 'object' and features[col].map(lambda x: isinstance(x, str)).all(),
             'Expected categorical strings: ' + col)
    need(str(details.eligible.dtype) == 'bool'
         and np.array_equal(features.addr_detail_eligible.to_numpy(), details.eligible.to_numpy().astype('float32')),
         'Eligibility explanations differ from feature flag')
    need(details.reason.map(lambda x: isinstance(x, str) and bool(x)).all(), 'Reason missing')
    eligible = details.eligible
    need(details.loc[eligible, 'main_number'].map(lambda n: isinstance(n, (int, np.integer)) and not isinstance(n, (bool, np.bool_))).all(),
         'Eligible rows require an integer main number')
    need(features.loc[~eligible, FEATURE_COLUMNS[:2]].eq('__MISSING__').all().all(), 'Rejected row exposes a key')


def ratio(count, total):
    return {'rows': int(count), 'denominator_rows': int(total), 'fraction': float(count / total) if total else None}


def quantiles(numbers):
    if not numbers:
        return {str(q): None for q in QUANTILES}
    # Exact integer order statistics avoid imposing an extra numeric cap.
    values = sorted(int(n) for n in numbers)
    return {str(q): values[int(np.floor(q * (len(values) - 1)))] for q in QUANTILES}



def summary(features, details):
    eligible = details.eligible
    n, e = len(features), int(eligible.sum())
    reasons = [{'reason': str(reason), **ratio(count, n)}
               for reason, count in details.reason.value_counts().sort_index().items()]
    detail = {}
    for name, flag in [('with_lane', details.lane.ne('')), ('with_alley', details.alley.ne('')),
                       ('with_explicit_house_suffix', details.house_has_suffix)]:
        detail[name] = {'eligible_detail_as_fraction_of_all_rows': ratio((flag & eligible).sum(), n),
                        'eligible_rows': ratio((flag & eligible).sum(), e),
                        'scope': 'new-guard eligible details only; not unfiltered raw-address occurrence'}
    return {'rows': n, 'eligible': ratio(e, n), 'reasons': reasons,
            'eligible_key_cardinality': {col: int(features.loc[eligible, col].nunique()) for col in FEATURE_COLUMNS[:2]},
            'detail_presence': detail,
            'eligible_main_number_quantiles': quantiles(details.loc[eligible, 'main_number'].tolist())}


def support(reference_features, reference_details, query_features, query_details):
    r_ok, q_ok = reference_details.eligible, query_details.eligible
    qn = int(q_ok.sum())
    result = {'reference_eligible_rows': int(r_ok.sum()), 'query_eligible_rows': qn,
              'denominator_excludes_ineligible_rows': True, 'keys': {}}
    for col in FEATURE_COLUMNS[:2]:
        counts = reference_features.loc[r_ok, col].value_counts()
        q_counts = query_features.loc[q_ok, col].map(counts).fillna(0).astype('int64')
        result['keys'][col] = {'reference_distinct_keys': len(counts),
            'query_distinct_keys': int(query_features.loc[q_ok, col].nunique()),
            'seen': ratio(q_counts.gt(0).sum(), qn), 'unseen': ratio(q_counts.eq(0).sum(), qn),
            'supported_by_at_least_10_fit_rows': ratio(q_counts.ge(10).sum(), qn),
            'reference_keys_with_at_least_10_rows': int(counts.ge(10).sum()),
            'reference_rows_in_keys_with_at_least_10_rows': ratio(counts[counts.ge(10)].sum(), int(r_ok.sum()))}
    blocks = pd.DataFrame({'block': reference_features.loc[r_ok, 'addr_house_block100'].to_numpy(),
                           'main': reference_details.loc[r_ok, 'main_number'].to_numpy()})
    distinct = blocks.groupby('block', sort=False).main.nunique()
    q_main_counts = query_features.loc[q_ok, 'addr_house_block100'].map(distinct).fillna(0).astype('int64')
    block_rows = blocks.block.value_counts()
    q_block_rows = query_features.loc[q_ok, 'addr_house_block100'].map(block_rows).fillna(0).astype('int64')
    result['block_with_at_least_2_distinct_fit_main_numbers'] = ratio(q_main_counts.ge(2).sum(), qn)
    result['block_with_both_10_fit_rows_and_2_distinct_main_numbers'] = ratio((q_main_counts.ge(2) & q_block_rows.ge(10)).sum(), qn)
    result['reference_blocks_with_at_least_2_distinct_main_numbers'] = int(distinct.ge(2).sum())
    result['distinct_main_numbers_are_not_distinct_projects'] = True
    return result


def deterministic_sample(details):
    selected = []
    def add(positions):
        for pos in positions:
            if int(pos) not in selected:
                selected.append(int(pos))
    for reason in sorted(details.reason.unique()):
        add(np.flatnonzero(details.reason.eq(reason).to_numpy())[:2])
    for mask in [details.eligible, details.eligible & details.lane.ne(''), details.eligible & details.alley.ne(''),
                 details.eligible & details.house_has_suffix]:
        add(np.flatnonzero(mask.to_numpy())[:2])
    if len(details):
        add(np.linspace(0, len(details) - 1, min(64, len(details)), dtype='int64'))
    return np.asarray(selected[:128], dtype='int64')


def reconstruct_check(frame, features, details, module, text, cat):
    chosen = deterministic_sample(details)
    sample = frame.iloc[chosen]
    X = pd.DataFrame(index=sample.index)
    X['address_raw'] = cat(text(sample.source_address).str.replace('台', '臺', regex=False))
    X['district'] = cat(sample.source_district)
    before = X.copy(deep=True)
    rebuilt = module.features_from_X(X)
    pd.testing.assert_frame_equal(rebuilt, features.iloc[chosen], check_exact=True)
    pd.testing.assert_frame_equal(X, before, check_exact=True)
    return {'rows': len(chosen), 'source_row_indices': sample.source_row_index.tolist(),
            'feature_values_dtypes_index_exact': True, 'query_input_unchanged': True,
            'original_87_cache_loaded': False, 'two_X_columns_reconstructed_from_original_provenance': True}


def examples(populations):
    result = []
    seen = set()
    # Fixed order and lexical reason order; no model error/price selection.
    for population in ('development_fit', 'development_query', 'valid', 'test'):
        frame, features, details = populations[population]
        population_start = len(result)
        for reason in sorted(details.reason.unique()):
            if len(result) - population_start >= 10:
                break
            positions = np.flatnonzero(details.reason.eq(reason).to_numpy())[:2]
            for pos in positions:
                if len(result) - population_start >= 10:
                    break
                row, detail = frame.iloc[pos], details.iloc[pos]
                split = 'train' if population.startswith('development_') else population
                key = (split, int(row.source_row_index))
                if key in seen:
                    continue
                seen.add(key)
                result.append({'population': population, 'split': split,
                    'source_row_index': int(row.source_row_index), 'source_excel_row': int(row.source_excel_row),
                    'address': None if pd.isna(row.source_address) else str(row.source_address),
                    'supplied_district': None if pd.isna(row.source_district) else str(row.source_district),
                    'eligible': bool(detail.eligible), 'reason': str(detail.reason),
                    'main_number': int(detail.main_number) if bool(detail.eligible) else None,
                    'addr_lane_path': features.iloc[pos].addr_lane_path,
                    'addr_house_block100': features.iloc[pos].addr_house_block100})
                if len(result) == 40:
                    return result
    return result


def render(value):
    lines = ['# R25 純 X 地址支持盤點', '',
        '本次只使用既有地址解析檔與凍結索引，沒有讀取價格快取、模型或預測，也沒有訓練或計算 MAE。兩類別欄與資格旗標、100 號桶均在盘點前固定；支持數只作描述，不是篩選門檻。', '',
        '原始 train 455,787 列保留 3 列夾入表頭；先映射至 455,784 列清理快取，再套原先保存的 529 列去重保留索引，得到 canonical 455,255 列。去重歷史規則曾包含價格，本次只沿用索引，沒有重算。valid/test 保留 11,994/47,073 列，不做全量合併或再次去重。', '',
        '| 群組 | 列数 | 合資格列 | 比率 | 巷弄路径唯一數 | 門牌桶唯一數 |',
        '|---|---:|---:|---:|---:|---:|']
    for name, item in value['populations'].items():
        lines.append(f"| {name} | {item['rows']:,} | {item['eligible']['rows']:,} | {item['eligible']['fraction']:.2%} | {item['eligible_key_cardinality']['addr_lane_path']:,} | {item['eligible_key_cardinality']['addr_house_block100']:,} |")
    lines += ['', '| X 參考→查詢 | path 見過率 | block 見過率 | block ≥10參考列 | block ≥2不同主號 |',
              '|---|---:|---:|---:|---:|']
    def percent(item):
        return '無合資格查詢列' if item['fraction'] is None else f"{item['fraction']:.2%}"
    for name, item in value['support'].items():
        keys = item['keys']
        lines.append(f"| {name} | {percent(keys['addr_lane_path']['seen'])} | {percent(keys['addr_house_block100']['seen'])} | {percent(keys['addr_house_block100']['supported_by_at_least_10_fit_rows'])} | {percent(item['block_with_at_least_2_distinct_fit_main_numbers'])} |")
    lines += ['', '比率分母為合資格查詢列，缺失/拒絕列不當作未見 key。同主號／同桶不代表同一棟或鄰近距離；至少兩種主號更不等於兩個建案。重複交易仍會增加列支持數。地址編碼可能不連續，也不能拿跨道路號碼差當距離。', '',
        '資格拒絕原因、合資格列的巷／弄／明確附號比例、主號經驗分位數（lower order statistic）與最多 40 個固定分層語法例，均見完整 JSON；沒有按價格誤差挑樣本。小樣本以原地址與行政區還原兩個原 X 欄，確認 features_from_X 與保存 parsed 的三個特徵值、dtype 和索引完全一致。這只是語法接線核對，不是定位準確率或價格效果。', '',
        'development query X 與 valid/test X 已被本研究查看，不能稱為未接觸的盲測；未使用其價格。後續價格候選是否有效尚未知。', '',
        '[完整盤點 JSON](../outputs/round25_research/address_support.json) · [SHA 收據](../outputs/round25_research/address_support.sha256) · [可重跑程式](../src/round25_support_audit.py)', '',
        f"程式 SHA：`{value['source_sha256']['src/round25_support_audit.py']}`；特徵規則來源 SHA：`{FEATURE_SOURCE_SHA256}`。輸入、來源、前後位元身分與執行時版本均記錄於 JSON。", '']
    return '\n'.join(lines)


def run():
    destinations = (JSON_PATH, JSON_PATH.with_suffix('.sha256'), MD_PATH)
    need(all(not path.exists() for path in destinations), 'Refusing to overwrite an existing/partial inventory')
    need(FEATURE_SOURCE_SHA256 != 'PENDING_FINAL_FEATURE_SOURCE', 'Feature source not frozen')
    bindings = Bindings()
    snapshot_name = 'runs/v3/snapshot_manifest.json'
    snapshot = bindings.json(snapshot_name, SNAPSHOT_SHA256, receipt=True)
    known = snapshot_paths(snapshot)
    manifest_name = 'outputs/round6/geography/development/manifest.json'
    manifest = bindings.json(manifest_name, known[manifest_name], receipt=True)
    for path in (manifest_name, str(Path(manifest_name).with_suffix('.sha256')).replace('\\', '/')):
        bindings.bind(path, known[path])
    need(manifest['stage'] == 'development' and manifest['geography_provenance_chain']['verified'] is True,
         'Geographic provenance/stage differs')
    audit_name = 'outputs/geography/address_audit/audit.json'
    audit = bindings.json(audit_name, known[audit_name])
    need(manifest['source_sha256'][audit_name] == known[audit_name], 'Parser audit not bound to manifest')
    parser_name = 'src/geo_address_audit.py'
    need(audit['source_sha256'] == known[parser_name] == manifest['source_sha256'][parser_name], 'Parser source identity differs')
    source_names = ['src/round25_support_audit.py', 'src/round25_address_features.py', parser_name, 'src/house_features.py']
    bindings.bind(source_names[0])
    bindings.bind(source_names[1], FEATURE_SOURCE_SHA256)
    bindings.bind(parser_name, audit['source_sha256'])
    bindings.bind('src/house_features.py', manifest['source_sha256']['src/house_features.py'])
    bindings.bind('reports/round25_research_plan.md')
    # All local source identities are checked before importing pure parsers.
    import round25_address_features as module
    import geo_address_audit as parser
    from house_features import _text, _cat
    plan_name = 'outputs/round6/candidate_plan.json'
    plan = bindings.json(plan_name, known[plan_name], receipt=True)
    need(plan['splits']['development'] == ['2022-08-11', '2023-08-10'], 'Development dates differ')
    index_names = {
        'canonical_keep': 'outputs/round6/geography/development/train_source_cache_indices.npy',
        'fit': 'outputs/round6/models/development/geography_catboost/fit_indices.npy',
        'query': 'outputs/round6/models/development/geography_catboost/query_indices.npy'}
    inputs = {}
    indices = {}
    for key, name in index_names.items():
        expected = manifest['index_files_sha256'][name] if key == 'canonical_keep' else known[name]
        need(known[name] == expected, 'Index manifest/snapshot mismatch')
        inputs[key] = bindings.bind(name, expected)
        indices[key] = np.load(ROOT / name, allow_pickle=False)
    keep, fi, qi = (indices[key] for key in ('canonical_keep', 'fit', 'query'))
    check_index(keep, 455784, 455255, 'canonical keep')
    split_contract = plan['split_contract']['development']
    for key, values, count in [('fit', fi, 240316), ('query', qi, 81234)]:
        check_index(values, 455255, count, key)
        need(len(values) == split_contract[key + '_rows'] and
             hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest() == split_contract[key + '_indices_content_sha256'],
             'Frozen split content differs: ' + key)
    entries = {item['split']: item for item in audit['splits']}
    populations, alignment, reconstruction = {}, {}, {}
    for split in SPLITS:
        entry = entries[split]
        name = entry['parsed_path'].replace('\\', '/')
        need(name == f'outputs/geography/address_audit/{split}_parsed.pkl'
             and entry['price_column_loaded'] is False and entry['post_training_dedup_applied'] is False,
             'Unexpected geographic input policy')
        expected = entry['parsed_sha256']
        need(expected == known[name] == manifest['source_sha256'][name], 'Parsed file authority differs')
        inputs[split + '_parsed'] = bindings.bind(name, expected)
        parsed = pd.read_pickle(ROOT / name)
        clean = check_parsed(parsed, entry, parser)
        expected_rows = {'train': 455784, 'valid': 11994, 'test': 47073}[split]
        need(len(clean) == expected_rows, 'Unexpected clean-cache row count')
        if split == 'train':
            canonical_positions(clean.existing_feature_cache_row_index.to_numpy(), keep, fi, qi)
            frame = clean.iloc[keep].reset_index(drop=True)
            name = 'canonical_train'
        else:
            frame, name = clean, split
        features, details = module.features_from_parsed(frame), module.eligibility_details(frame)
        check_features(frame, features, details)
        populations[name] = (frame, features, details)
        reconstruction[split] = reconstruct_check(frame, features, details, module, _text, _cat)
        alignment[split] = {'original_rows': len(parsed), 'embedded_headers_removed': len(parsed) - len(clean),
                            'cache_rows': len(clean), 'canonical_rows': len(frame),
                            'saved_canonical_duplicates_removed': len(clean) - len(frame),
                            'full_schema_and_provenance_checked': True, 'target_based_dedup_recomputed': False}
        if split == 'train':
            for name, positions in [('development_fit', fi), ('development_query', qi)]:
                populations[name] = tuple(obj.iloc[positions].reset_index(drop=True) for obj in (frame, features, details))
        del parsed, clean
    summary_values = {name: summary(values[1], values[2]) for name, values in populations.items()}
    supports = {}
    for label, reference, query in [('development_fit_to_query', 'development_fit', 'development_query'),
                                  ('canonical_train_to_test', 'canonical_train', 'test')]:
        supports[label] = support(*populations[reference][1:], *populations[query][1:])
    examples_value = examples(populations)
    before = dict(bindings.files)
    after = bindings.verify_unchanged()
    source_map = {name: before[name]['sha256'] for name in source_names}
    evidence_map = {name: item['sha256'] for name, item in before.items()
                    if name not in source_map and name not in {item['path'] for item in inputs.values()}}
    value = {'round': 25, 'status': 'pure_X_support_inventory_complete',
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source_sha256': source_map, 'evidence_sha256': evidence_map, 'inputs': inputs,
        'input_before': before, 'input_after': after, 'all_read_sources_and_inputs_unchanged': before == after,
        'audit_rules': AUDIT_RULES, 'feature_rules': module.RULES,
        'runtime_versions': {'python': platform.python_version(), 'numpy': np.__version__, 'pandas': pd.__version__},
        'read_scope': {'parsed_splits_loaded': list(SPLITS), 'original87_features_loaded': False,
            'price_cache_or_excel_loaded': False, 'model_or_prediction_loaded': False,
            'labels_used': False, 'model_training_or_price_scoring': False,
            'canonical_dedup_recomputed': False, 'existing_target_containing_dedup_index_reused': True,
            'query_X_observed': True, 'historically_untouched_blind_X': False,
            'raw_ID_used_only_as_provenance': True, 'external_answers_used': False},
        'alignment': alignment, 'development_contract': {'dates': plan['splits']['development'], **split_contract},
        'populations': summary_values, 'support': supports, 'reconstruction_checks': reconstruction,
        'syntax_examples': examples_value,
        'limitations': ['Support rows are not unique projects or physical neighbours.',
                       'Address-number differences do not establish geographic distances.',
                       'Eligibility is parser evidence, not verified address accuracy.',
                       'No bins, rules or model settings were selected using this inventory.',
                       'No predictive improvement or price-model acceptance is established.']}
    text = json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + '\n'
    markdown = render(value)
    digest = hashlib.sha256(text.encode('utf-8')).hexdigest()
    for path, contents in [(JSON_PATH, text), (JSON_PATH.with_suffix('.sha256'), digest + '\n'), (MD_PATH, markdown)]:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex + '.tmp')
        with temporary.open('x', encoding='utf-8', newline='\n') as handle:
            handle.write(contents)
        need(not path.exists(), 'Destination appeared during publication')
        temporary.rename(path)
    print(json.dumps({'status': value['status'], 'json_sha256': sha(JSON_PATH), 'markdown_sha256': sha(MD_PATH),
                      'eligible_rows': {name: item['eligible']['rows'] for name, item in summary_values.items()}}, ensure_ascii=False))


def self_check():
    checks = []
    keep = np.array([0, 2, 3, 5], dtype='int64')
    fi, qi = np.array([0, 1], dtype='int64'), np.array([2, 3], dtype='int64')
    left, right = canonical_positions(np.arange(6, dtype='int64'), keep, fi, qi)
    need(left.tolist() == [0, 2] and right.tolist() == [3, 5], 'Synthetic mapping failed')
    checks.append('canonical_then_stage_composition')
    for label, action in [
        ('duplicate_cache_index', lambda: canonical_positions(np.array([0, 1, 1], dtype='int64'), keep, fi, qi)),
        ('float_indices', lambda: check_index(keep.astype('float64'), 6, 4, 'synthetic')),
        ('out_of_bounds', lambda: check_index(np.array([0, 6], dtype='int64'), 6, 2, 'synthetic')),
        ('fit_query_overlap', lambda: canonical_positions(np.arange(6, dtype='int64'), keep, fi, fi))]:
        try:
            action()
        except RuntimeError:
            checks.append(label + '_rejected')
        else:
            raise AssertionError(label)
    return {'status': 'passed', 'checks': checks, 'data_or_model_read': False, 'price_scoring': False}


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    choice = cli.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run', action='store_true')
    choice.add_argument('--self-check', action='store_true')
    args = cli.parse_args()
    if args.self_check:
        print(json.dumps(self_check()))
    else:
        run()
