"""R26 target-free road-pair support inventory. --run requires prior authorization.
Load only three saved parsed geographic frames and frozen canonical/split indices.
The fixed one-column parser is never tuned using this inventory or prices.
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
import re
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round26_research'
JSON_PATH = OUT / 'address_support.json'
MD_PATH = ROOT / 'reports/round26_address_support.md'
SPLITS = ('train', 'valid', 'test')
FEATURE_COLUMNS = ['addr_explicit_road_pair']
FEATURE_SOURCE_SHA256 = '65f5268d46fc9677408b7224ad1838ff46a41800f064ee0f4ffde6eb187ce834'
NUMBER_SOURCE_SHA256 = 'deb3be7732bc5fd1be0a117afc9d762fc374e8144be35def35fa85bd41e511de'
RESEARCH_PLAN_SHA256 = 'b6005477ac4846c03d7ff66f2f3b8176855005a8e93461c5eea5f24935159b75'
SNAPSHOT_SHA256 = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
AUDIT_RULES = {
    'research_only': True, 'learned_vocabulary_or_target_statistics': False,
    'support_reference_pairs': ['development_fit_to_query', 'canonical_train_to_test'],
    'minimum_reference_rows_diagnostic': 10,
    'minimum_distinct_normalized_address_strings_diagnostic': 2,
    'support_thresholds_do_not_filter_features': True,
    'query_support_denominators': ['eligible_query_rows', 'all_query_rows'],
    'ineligible_rows_are_not_unseen_keys': True,
    'section_presence': 'at least one eligible canonical road name ends in Arabic digits plus 段',
    'example_maximum_total': 40, 'example_maximum_per_population': 10,
    'example_eligible_maximum_per_population': 2,
    'example_rejected_order': 'descending population reason count, then ascending reason name',
    'example_maximum_per_rejection_reason': 2,
    'reconstruction_maximum_per_split': 128,
    'reconstruction_selection': 'first two per reason, first two eligible section rows, plus 64 evenly spaced positions',
    'distinct_address_strings_are_not_distinct_projects_or_locations': True,
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

def ratio(count, total):
    return {'rows': int(count), 'denominator_rows': int(total), 'fraction': float(count / total) if total else None}

def check_features(frame, features, details):
    need(list(features.columns) == FEATURE_COLUMNS and features.index.equals(frame.index)
         and list(details.columns) == ['eligible','reason','roads','key']
         and details.index.equals(frame.index), 'Feature/details alignment differs')
    feature = features[FEATURE_COLUMNS[0]]
    need(str(feature.dtype) == 'object' and feature.map(lambda x:isinstance(x,str)).all(), 'Expected categorical strings')
    need(str(details.eligible.dtype) == 'bool' and feature.equals(details.key), 'Key/details differ')
    need(details.reason.map(lambda x:isinstance(x,str) and bool(x)).all(), 'Reason missing')
    valid = details.eligible
    need(details.loc[valid,'roads'].map(lambda r:isinstance(r,tuple) and len(r)==2
         and all(isinstance(x,str) and x for x in r) and r[0]<r[1]).all(), 'Eligible roads must be two sorted distinct names')
    need(details.loc[~valid,'roads'].map(lambda r:isinstance(r,tuple) and r==()).all()
         and feature.loc[~valid].eq('__MISSING__').all()
         and feature.loc[valid].ne('__MISSING__').all(), 'Rejected/eligible keys inconsistent')


def has_section(roads):
    return any(re.search(r'[0-9]+段$', road) is not None for road in roads)


def summary(features, details):
    n, eligible = len(features), details.eligible
    e = int(eligible.sum())
    count = int(details.loc[eligible,'roads'].map(has_section).sum())
    return {'rows':n, 'eligible':ratio(e,n),
        'reasons':[{'reason':str(reason), **ratio(rows,n)}
                   for reason,rows in details.reason.value_counts().sort_index().items()],
        'eligible_key_cardinality':int(features.loc[eligible,FEATURE_COLUMNS[0]].nunique()),
        'eligible_with_section':{'eligible_rows':ratio(count,e), 'all_rows':ratio(count,n)},
        'section_presence_scope':AUDIT_RULES['section_presence']}


def support(reference_frame, reference_features, reference_details, query_frame, query_features, query_details):
    r_ok, q_ok = reference_details.eligible, query_details.eligible
    nr, nq, total = int(r_ok.sum()), int(q_ok.sum()), len(query_frame)
    rk = reference_features.loc[r_ok,FEATURE_COLUMNS[0]]
    qk = query_features.loc[q_ok,FEATURE_COLUMNS[0]]
    counts = rk.value_counts()
    addresses = pd.DataFrame({'key':rk.to_numpy(),
        'address':reference_frame.loc[r_ok,'address_normalized'].to_numpy()})
    distinct = addresses.groupby('key',sort=False).address.nunique()
    q_counts = qk.map(counts).fillna(0).astype('int64')
    q_distinct = qk.map(distinct).fillna(0).astype('int64')
    def denominators(count):
        return {'eligible_query_rows':ratio(count,nq), 'all_query_rows':ratio(count,total)}
    return {'reference_eligible_rows':nr, 'query_eligible_rows':nq, 'all_query_rows':total,
        'reference_distinct_keys':int(len(counts)), 'query_distinct_keys':int(qk.nunique()),
        'seen':denominators(q_counts.gt(0).sum()), 'unseen':denominators(q_counts.eq(0).sum()),
        'supported_by_at_least_10_fit_rows':denominators(q_counts.ge(10).sum()),
        'supported_by_at_least_2_distinct_fit_address_texts':denominators(q_distinct.ge(2).sum()),
        'supported_by_both_10_rows_and_2_address_texts':denominators((q_counts.ge(10)&q_distinct.ge(2)).sum()),
        'reference_keys_with_at_least_10_rows':int(counts.ge(10).sum()),
        'reference_keys_with_at_least_2_address_texts':int(distinct.ge(2).sum()),
        'ineligible_queries_not_counted_as_unseen':True,
        'distinct_address_texts_are_not_distinct_projects_or_physical_locations':True}


def deterministic_sample(details):
    positions = []
    for reason in sorted(details.reason.unique()):
        positions.extend(np.flatnonzero(details.reason.eq(reason).to_numpy())[:2].tolist())
    section = details.eligible & details.roads.map(has_section)
    positions.extend(np.flatnonzero(section.to_numpy())[:2].tolist())
    if len(details):
        positions.extend(np.linspace(0,len(details)-1,min(64,len(details)),dtype='int64').tolist())
    return np.asarray(list(dict.fromkeys(positions))[:128],dtype='int64')


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
    result, coverage = [], {}
    for population in ('development_fit','development_query','valid','test'):
        frame, features, details = populations[population]
        counts = details.loc[~details.eligible,'reason'].value_counts()
        rejection_order = sorted(counts.index, key=lambda name:(-int(counts[name]),str(name)))
        selected, chosen_reasons = [], []
        selected.extend(np.flatnonzero(details.eligible.to_numpy())[:2].tolist())
        for reason in rejection_order:
            remaining = 10-len(selected)
            if remaining <= 0: break
            candidates = np.flatnonzero(details.reason.eq(reason).to_numpy())[:min(2,remaining)].tolist()
            if candidates:
                selected.extend(candidates); chosen_reasons.append(str(reason))
        for position in selected:
            row, detail = frame.iloc[position], details.iloc[position]
            result.append({'population':population,
                'split':'train' if population.startswith('development_') else population,
                'source_row_index':int(row.source_row_index), 'source_excel_row':int(row.source_excel_row),
                'address':None if pd.isna(row.source_address) else str(row.source_address),
                'supplied_district':None if pd.isna(row.source_district) else str(row.source_district),
                'eligible':bool(detail.eligible), 'reason':str(detail.reason),
                'roads':list(detail.roads), 'addr_explicit_road_pair':features.iloc[position,0]})
        coverage[population] = {'example_count':len(selected),
            'eligible_example_count':int(details.iloc[selected].eligible.sum()),
            'covered_rejection_reasons':chosen_reasons,
            'uncovered_rejection_reasons':[{'reason':str(name),'rows':int(counts[name])}
                for name in rejection_order if str(name) not in chosen_reasons],
            'selection_is_frequency_stratified_not_random':True}
    need(len(result)<=40, 'Example budget exceeded')
    return result, coverage


def render(value):
    lines = ['# R26 純 X 明示雙道路支持盤點','',
        '只讀既有 target-free 地址 parsed 與保存 K/fi/qi，沒有價格快取、模型、預測、MAE或新訓練。唯一道路對類別與嚴格語法在盤點前固定；10列／2種原文只作診斷，不篩選特徵。','',
        'train原始455,787列先移除3列夾入表頭，再依existing cache index對齊455,784列，沿保存去重索引得到455,255列。歷史去重曾包含target，本次未重算；development fit/query為240,316/81,234列。valid/test為11,994/47,073列，不合併或再去重。','',
        '| 群組 | 全列數 | 合資格列 | 合資格率 | 完整pair唯一數 | 含段占合資格率 |',
        '|---|---:|---:|---:|---:|---:|']
    def pct(item): return '無分母' if item['fraction'] is None else f"{item['fraction']:.2%}"
    for name,item in value['populations'].items():
        lines.append(f"| {name} | {item['rows']:,} | {item['eligible']['rows']:,} | {pct(item['eligible'])} | {item['eligible_key_cardinality']:,} | {pct(item['eligible_with_section']['eligible_rows'])} |")
    lines += ['', '| X參考→查詢 | 指標 | 列數 | 占合資格query | 占全部query |', '|---|---|---:|---:|---:|']
    for name,item in value['support'].items():
        for key,label in [('seen','見過key'),('supported_by_at_least_10_fit_rows','至少10參考列'),
                          ('supported_by_at_least_2_distinct_fit_address_texts','至少2種normalized原文')]:
            metric=item[key]
            lines.append(f"| {name} | {label} | {metric['eligible_query_rows']['rows']:,} | {pct(metric['eligible_query_rows'])} | {pct(metric['all_query_rows'])} |")
    lines += ['', '缺失／拒絕列不算未見key，所有支持指標同時保存兩種分母。列數可能來自同案重複交易；兩種normalized文字不代表不同建案或真實位置。固定key只證明文字語法關係，不證明道路實際相交、住宅定位或距離。','',
        '每分區最多10例，先保留最多2個合資格例，其餘按拒絕原因列數降序、同數按名稱排序，每類最多2例。JSON明列因40例預算未覆蓋的原因；這不是隨機代表性樣本，也沒有依價格挑例。兩個原X欄的固定少量重建核对不載87欄快取。','',
        'query及valid/test X已被查看，不能稱未接觸的盲測；任何價格效果或後續候選採用均未由本盤點建立。','',
        '[完整JSON](../outputs/round26_research/address_support.json) · [SHA收據](../outputs/round26_research/address_support.sha256) · [盤點程式](../src/round26_support_audit.py)','',
        f"來源SHA：`{value['source_sha256']['src/round26_support_audit.py']}`；固定特徵模組：`{FEATURE_SOURCE_SHA256}`。"]
    return '\n'.join(lines)+'\n'


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
    source_names = ['src/round26_support_audit.py', 'src/round26_intersection_features.py', parser_name, 'src/house_features.py', 'src/round25_address_features.py']
    bindings.bind(source_names[0])
    bindings.bind(source_names[1], FEATURE_SOURCE_SHA256)
    bindings.bind(parser_name, audit['source_sha256'])
    bindings.bind('src/house_features.py', manifest['source_sha256']['src/house_features.py'])
    bindings.bind('src/round25_address_features.py', NUMBER_SOURCE_SHA256)
    bindings.bind('reports/round26_research_plan.md', RESEARCH_PLAN_SHA256)
    # All local source identities are checked before importing pure parsers.
    import round26_intersection_features as module
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
        supports[label] = support(*populations[reference], *populations[query])
    examples_value, example_coverage = examples(populations)
    before = dict(bindings.files)
    after = bindings.verify_unchanged()
    source_map = {name: before[name]['sha256'] for name in source_names}
    evidence_map = {name: item['sha256'] for name, item in before.items()
                    if name not in source_map and name not in {item['path'] for item in inputs.values()}}
    value = {'round': 26, 'status': 'pure_X_support_inventory_complete',
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
        'syntax_examples': examples_value, 'example_coverage': example_coverage,
        'limitations': ['Support rows are not unique projects or physical neighbours.',
                       'Distinct normalized address text does not establish different projects or physical positions.',
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
    keep=np.array([0,2,3,5],dtype='int64'); fi=np.array([0,1],dtype='int64'); qi=np.array([2,3],dtype='int64')
    left,right=canonical_positions(np.arange(6,dtype='int64'),keep,fi,qi)
    need(left.tolist()==[0,2] and right.tolist()==[3,5], 'Canonical-to-stage mapping differs')
    return {'status':'passed','checks':['canonical_then_stage_composition'],
            'data_or_model_read':False,'price_scoring':False}


if __name__ == '__main__':
    cli=argparse.ArgumentParser(description=__doc__)
    choice=cli.add_mutually_exclusive_group(required=True)
    choice.add_argument('--run',action='store_true'); choice.add_argument('--self-check',action='store_true')
    args=cli.parse_args()
    if args.self_check: print(json.dumps(self_check()))
    else: run()
