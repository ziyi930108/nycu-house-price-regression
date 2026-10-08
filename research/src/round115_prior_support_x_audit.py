"""R115 pure-X project-fold versus transaction-fold support exposure audit.

No prices, NPZ, model, prior-state, reference/spatial state, fit or scorer.
Preparation is not execution authorization: independent metadata/source
preflight is mandatory before --freeze and --run. Outputs are aggregates.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PLAN = 'reports/round115_plan.md'
OUT = 'outputs/round115'
PREFLIGHT = 'outputs/research_round115_20261008/preflight_review.json'
STAGES = ('development', 'confirmation', 'auxiliary', 'offset_2023_02_11')
WINDOWS = {
    'development': ('2022-08-11', '2023-08-10', 240316, 81234),
    'confirmation': ('2023-08-11', '2024-08-10', 321550, 133697),
    'auxiliary': ('2021-08-11', '2022-08-10', 134805, 105511),
    'offset_2023_02_11': ('2023-02-11', '2024-02-10', 272117, 107169),
}
LEVELS = (('address_city', 'address_district', 'transaction_type', 'building_type'),
          ('address_city', 'transaction_type', 'building_type'),
          ('transaction_type', 'building_type'), ('transaction_type',))
REQUIRED = set(sum(LEVELS, ())) | {'project_name', 'address_raw', 'address_road', 'address_section',
                                       'trade_day_index', 'trade_date_invalid', 'trade_date_out_of_range'}
FORBIDDEN = {'總價元', '編號', 'target', 'y', 'ID', 'id', 'train_y_million', 'query_y'}
AUTHORITIES = {
    'outputs/round48/plan.json': 'ece3c43bae1d2f4c5f471d623a0a1082dcd8152dc3b0d20e91e7bc10c1a184af',
    'outputs/round60/results.json': '8e1dc9e35e1685f4169327016b477fd21ff1b46fadf83cfa8cfb915bd23f1934',
    'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'outputs/round90/preparation/manifest.json': '400f26bc74c00f7fa0e8062ceb509872eeb6a1af3c3b3437512569c441d17fa7',
    'outputs/round97/nn_prepared/prepared.json': '3496d144e8ca6762d10fa786d5968eaa3ee3afba6d4513f58107b6f2fd31b24a',
}
REFERENCE_SOURCES = ('src/round52_price_prior.py', 'src/round60_recency_price_prior.py',
                     'src/v3_cluster_stability.py')
FEATURES = 'data_cache/train_features.pkl'
MAPPING = 'outputs/round6/geography/development/train_source_cache_indices.npy'
SCHEMA = 'reports/feature_schema.json'
QUANTILES = (0., .1, .25, .5, .75, .9, .99, 1.)


def need(condition, message):
    if not condition:
        raise ValueError(message)


def safe(name):
    p = (ROOT / name).resolve()
    need(p.is_relative_to(ROOT), 'Workspace-relative artifact required')
    return p


def sha(name):
    h = hashlib.sha256()
    with safe(name).open('rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def values_sha(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def identity(a):
    return {'dtype': str(a.dtype), 'shape': list(a.shape), 'values_sha256': values_sha(a)}


def frame_identity(X):
    return {'rows': len(X), 'columns': list(X.columns), 'dtypes': {c: str(X[c].dtype) for c in X},
            'values_sha256': values_sha(pd.util.hash_pandas_object(X, index=True).to_numpy(dtype='<u8'))}


def read_json(name):
    return json.loads(safe(name).read_text(encoding='utf-8-sig'))


def write(name, value):
    p = safe(name)
    with p.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        f.write('\n')
    with p.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(sha(name) + '\n')


def now():
    return datetime.now(timezone.utc).isoformat()


def record(name):
    return {'path': name, 'sha256': sha(name)}


def source_hashes():
    return {name: sha(name) for name in (str(Path(__file__).relative_to(ROOT)).replace('\\', '/'), *REFERENCE_SOURCES)}


def parameters():
    return {'round': 115, 'role': 'pure_X_fold_exposure_audit_not_model', 'stage_order': list(STAGES),
            'levels': [list(x) for x in LEVELS], 'minimum_raw_cell_rows': 30, 'global_minimum_rows': 1,
            'transaction_folds': 'for each origin: p=default_rng(42).permutation(n); folds[p]=arange(n)%5',
            'group_folds': 'original canonical cluster key sha256(42|key) first8 little-endian mod5',
            'date_weights': 'saved full outer-fit float32, converted to float64; never recomputed/renormalized',
            'project_mass_ESS': '(sum_g M_g)^2/sum_g(M_g^2); M_g=sum(saved date weights within reference cell/group)',
            'maximum_project_date_mass_share': 'max_g(M_g)/sum_g(M_g)',
            'project_grouping': 'original project/city/district cluster; missing original address/road/row fallbacks',
            'quantiles': list(QUANTILES), 'quantile_method': 'linear', 'new_fits': 0, 'GPU_calls': 0,
            'prices_or_query_labels': False, 'NPZ_allowed': False, 'new_CSV': False,
            'automatic_model_go_gate': False, 'threshold_or_fold_search': False}


def specification():
    """Read fixed metadata only; no real DataFrame/array deserialization or stats."""
    inputs = dict(AUTHORITIES)
    docs = {}
    for name, digest in AUTHORITIES.items():
        need(safe(name).is_file() and sha(name) == digest, 'Missing/changed fixed metadata authority: ' + name)
        docs[name] = read_json(name)
    p48, r60, r79, pure, neural = [docs[name] for name in AUTHORITIES]
    need(pure['status'] == 'prepared_target_free' and neural['query_labels_available'] is False,
         'Original target-free/fit-only provenance required')
    inputs.update({name: p48['input_sha256'][name] for name in (FEATURES, MAPPING, SCHEMA)})
    origins = {}
    for stage in STAGES[:3]:
        start, end, nf, nq = WINDOWS[stage]
        contract, a, b = p48['split_contract'][stage], r60['stages'][stage], r79['stages'][stage]
        need(p48['splits'][stage] == [start, end] and contract['fit_rows'] == nf and contract['query_rows'] == nq,
             'Original August outer split contract')
        paths = {kind: f'outputs/round60/{stage}/{kind}.npy' for kind in ('fit_indices', 'query_indices')}
        paths.update(group_folds=f'outputs/round60/{stage}/prior_fold_ids.npy',
                     weights=f'outputs/round79/{stage}/model/date_weights.npy')
        meta_path = f'outputs/round79/{stage}/model/metadata.json'
        inputs[meta_path] = b['artifact_sha256'][meta_path]
        need(sha(meta_path) == inputs[meta_path], 'Changed date-weight provenance metadata')
        meta = read_json(meta_path)
        prep = meta['R63_preparation']
        need(meta['fit_rows'] == prep['fit_rows'] == nf and meta['query_rows'] == prep['query_rows'] == nq
             and meta['fit_indices_content_sha256'] == contract['fit_indices_content_sha256']
             and meta['query_indices_content_sha256'] == contract['query_indices_content_sha256']
             and meta['date_weight_provenance_verified_in_driver'] is True
             and meta['date_weights_recomputed'] is False and meta['date_weights_renormalized'] is False
             and meta['fit_date_weights_sha256'] == prep['train_sample_weight_sha256'], 'R63 standalone weight identity/axis')
        for role in ('fit_indices', 'query_indices', 'group_folds'):
            inputs[paths[role]] = a['artifact_sha256'][paths[role]]
        inputs[paths['weights']] = b['artifact_sha256'][paths['weights']]
        origins[stage] = {'paths': paths, 'index_contract': contract, 'weights_values_sha256': meta['fit_date_weights_sha256'],
                          'metadata': meta_path, 'X_mode': 'canonical_from_cache', 'fit_rows': nf, 'query_rows': nq}
    stage = STAGES[3]
    need(pure['fit_rows'] == neural['fit_rows'] == WINDOWS[stage][2]
         and pure['query_rows'] == neural['query_rows'] == WINDOWS[stage][3], 'Original offset dimensions')
    paths = {role: pure['artifacts'][role]['path'] for role in ('Xfit', 'Xquery', 'fit_indices', 'query_indices')}
    need(paths == {role: f'outputs/round90/preparation/{role}.' + ('pkl' if role.startswith('X') else 'npy') for role in paths},
         'Only saved target-free offset frames and canonical indices allowed')
    paths.update(group_folds='outputs/round97/nn_prepared/fold_ids.npy',
                 weights='outputs/round97/nn_prepared/date_weights_float32.npy',
                 nn_source_indices='outputs/round97/nn_prepared/source_indices.npy')
    need(neural['fold_ids'] == paths['group_folds'] and neural['saved_date_weights'] == paths['weights']
         and neural['source_indices'] == paths['nn_source_indices']
         and neural['source_indices_contract']['not_raw_cache_indices'] is True
         and neural['source_indices_contract']['not_reset_fit_row_numbers'] is True, 'Offset source-index axis')
    for role in ('Xfit', 'Xquery', 'fit_indices', 'query_indices'):
        inputs[paths[role]] = pure['artifacts'][role]['sha256']
    for role in ('group_folds', 'weights', 'nn_source_indices'):
        inputs[paths[role]] = neural['artifact_sha256'][paths[role]]
    wi = neural['prior_arrays']['train_sample_weight']
    need(wi['dtype'] == 'float32' and wi['shape'] == [WINDOWS[stage][2]], 'Offset saved date weights schema')
    origins[stage] = {'paths': paths, 'X_mode': 'saved_target_free', 'fit_rows': WINDOWS[stage][2], 'query_rows': WINDOWS[stage][3],
                      'weights_values_sha256': wi['values_sha256'],
                      'index_contract': {kind + '_indices_content_sha256': pure['arrays'][kind + '_indices']['content_sha256']
                                         for kind in ('fit', 'query')},
                      'frames': {key: pure['frames'][key] for key in ('Xfit', 'Xquery')}}
    blockers = [name for name in inputs if not safe(name).is_file()]
    return {'parameters': parameters(), 'source_sha256': source_hashes(), 'plan': record(PLAN),
            'input_sha256': inputs, 'origins': origins, 'blockers': blockers,
            'real_X_or_arrays_deserialized': False, 'input_byte_hashes_verified': False,
            'missing_file_fallback_allowed': False}


def verify_hashes(mapping):
    for name, digest in mapping.items():
        need(safe(name).is_file() and sha(name) == digest, 'Missing/changed frozen readset file: ' + name)


def permit_artifact(name, specification):
    allowed = {FEATURES, MAPPING}
    for origin in specification['origins'].values():
        allowed.update(origin['paths'].values())
    need(name in allowed and name in specification['input_sha256'], 'Artifact outside target-free typed readset')
    need(safe(name).suffix in ('.npy', '.pkl'), 'Only standalone NPY or declared pure-X pickle; NPZ is prohibited')


def norm(s):
    v = s.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
    return v.mask(v.isin(['', 'nan', 'None', '<NA>', 'NaT']), '__MISSING__')


def keys(X):
    fields = {c: norm(X[c]).to_numpy() for c in set(sum(LEVELS, ()))}
    return [np.asarray([json.dumps(row, ensure_ascii=False, separators=(',', ':'))
                        for row in zip(*(fields[c] for c in level))], dtype=object) for level in LEVELS]


def clusters(X, indices):
    fields = {c: norm(X[c]).to_numpy() for c in ('project_name', 'address_city', 'address_district',
                                               'address_raw', 'address_road', 'address_section')}
    result, missing = [], np.zeros(len(X), bool)
    for i in range(len(X)):
        city, district, project = (fields[k][i] for k in ('address_city', 'address_district', 'project_name'))
        if project != '__MISSING__':
            key = ['project', city, district, project]
        elif fields['address_raw'][i] != '__MISSING__':
            missing[i] = True
            key = ['missing_address', city, district, fields['address_raw'][i]]
        elif fields['address_road'][i] != '__MISSING__':
            missing[i] = True
            key = ['missing_road', city, district, fields['address_road'][i], fields['address_section'][i]]
        else:
            missing[i] = True
            key = ['missing_row', int(indices[i])]
        result.append(json.dumps(key, ensure_ascii=False, separators=(',', ':')))
    return np.asarray(result, dtype=object), missing


def original_folds(groups):
    d = {g: int.from_bytes(hashlib.sha256(('42|' + g).encode('utf-8')).digest()[:8], 'little') % 5 for g in np.unique(groups)}
    return np.asarray([d[g] for g in groups], dtype=np.int64)


def query_status_masks(fit, query):
    """Exact original group_labels membership; preserve its pipe-joined keys."""
    fp, qp = norm(fit.project_name), norm(query.project_name)
    fk = norm(fit.address_city) + '|' + norm(fit.address_district) + '|' + fp
    qk = norm(query.address_city) + '|' + norm(query.address_district) + '|' + qp
    missing = qp.eq('__MISSING__').to_numpy()
    seen = (~missing) & qk.isin(set(fk[~fp.eq('__MISSING__')])).to_numpy()
    return {'seen': seen, 'new_nonmissing': (~missing) & ~seen, 'missing': missing}


def transaction_folds(n):
    need(type(n) is int and n >= 5, 'At least five fit rows')
    folds = np.empty(n, dtype=np.int64)
    folds[np.random.default_rng(42).permutation(n)] = np.arange(n, dtype=np.int64) % 5
    return folds


def check_X(X, columns=None):
    need(isinstance(X, pd.DataFrame) and X.columns.is_unique and X.index.equals(pd.RangeIndex(len(X)))
         and len(X) > 0 and REQUIRED <= set(X) and not FORBIDDEN.intersection(X), 'Target-free frame schema')
    if columns is not None:
        need(list(X.columns) == list(columns), 'Original feature-column order')


def check_index(a, n):
    need(isinstance(a, np.ndarray) and a.dtype == np.int64 and a.ndim == 1 and len(a) > 0
         and (a >= 0).all() and (a < n).all() and (np.diff(a) > 0).all(), 'Sorted unique canonical int64 indices')


def check_weights(w, n):
    need(isinstance(w, np.ndarray) and w.dtype == np.float32 and w.shape == (n,)
         and np.isfinite(w).all() and (w > 0).all(), 'Standalone saved positive float32 date weights required')


def describe(a):
    a = np.asarray(a, dtype=np.float64)
    need(a.ndim == 1 and np.isfinite(a).all(), 'Finite aggregate input')
    if not len(a):
        return {'rows': 0, 'mean': None, 'min': None, 'max': None, 'quantiles': {str(q): None for q in QUANTILES}}
    return {'rows': len(a), 'mean': float(a.mean()), 'min': float(a.min()), 'max': float(a.max()),
            'quantiles': {str(q): float(np.quantile(a, q, method='linear')) for q in QUANTILES}}


def group_table(cell, group, weight):
    need(len(cell) == len(group) == len(weight) and len(cell) > 0, 'Nonempty reference cell/group rows')
    rows = pd.DataFrame({'cell': cell, 'group': group, 'mass': np.asarray(weight, dtype=np.float64)})
    pairs = rows.groupby(['cell', 'group'], sort=True)['mass'].agg(['size', 'sum'])
    pairs['sum_squared'] = pairs['sum'] ** 2
    cells = pairs.groupby(level='cell', sort=True).agg(raw_rows=('size', 'sum'), project_groups=('size', 'size'),
        largest_group_rows=('size', 'max'), date_mass=('sum', 'sum'), squared_project_mass=('sum_squared', 'sum'),
        largest_group_mass=('sum', 'max'))
    cells['largest_project_row_share'] = cells['largest_group_rows'] / cells['raw_rows']
    cells['date_weight_project_ESS'] = cells['date_mass'] ** 2 / cells['squared_project_mass']
    cells['largest_project_date_mass_share'] = cells['largest_group_mass'] / cells['date_mass']
    return cells, pairs['size']


METRICS = ('raw_rows', 'project_groups', 'largest_project_row_share', 'date_weight_project_ESS',
           'largest_project_date_mass_share')


def route(fit_keys, fit_groups, weights, reference, query_keys, query_groups):
    """Pure count/mass router; all key strings stay inside memory."""
    n = len(query_groups)
    need(reference.dtype == bool and reference.shape == (len(fit_groups),) and reference.any()
         and len(fit_keys) == len(query_keys) == 4, 'Reference/router dimensions')
    global_table, global_pairs = group_table(np.zeros(int(reference.sum()), np.int8), fit_groups[reference], weights[reference])
    global_row = global_table.loc[0]
    level = np.full(n, 4, dtype=np.int8)
    data = {key: np.full(n, float(global_row[key]), np.float64) for key in METRICS}
    group_counts = pd.Series(fit_groups[reference]).value_counts(sort=False)
    peers = pd.Series(query_groups).map(group_counts).fillna(0).to_numpy(dtype=np.float64)
    selected_peers = peers.copy()
    all_counts = np.zeros((n, 4), np.int64)
    cell_summaries = []
    for j, (fk, qk) in enumerate(zip(fit_keys, query_keys)):
        table, pair_counts = group_table(fk[reference], fit_groups[reference], weights[reference])
        raw = pd.Series(qk).map(table['raw_rows']).fillna(0).to_numpy(dtype=np.int64)
        all_counts[:, j] = raw
        take = (level == 4) & (raw >= 30)
        if take.any():
            level[take] = j
            for key in METRICS:
                data[key][take] = pd.Series(qk[take]).map(table[key]).to_numpy(dtype=np.float64)
            ix = pd.MultiIndex.from_arrays([qk[take], query_groups[take]], names=['cell', 'group'])
            selected_peers[take] = pair_counts.reindex(ix, fill_value=0).to_numpy(dtype=np.float64)
        cell_summaries.append({'level': j, 'all_cells': {key: describe(table[key].to_numpy()) for key in METRICS},
                               'supported_cells': {key: describe(table.loc[table.raw_rows >= 30, key].to_numpy()) for key in METRICS}})
    data.update(same_cluster_peers_reference=peers, same_cluster_peers_selected_cell=selected_peers,
                same_cluster_selected_row_share=selected_peers / data['raw_rows'])
    need(all(np.isfinite(v).all() for v in data.values()) and (data['raw_rows'] > 0).all(), 'Finite positive selected support')
    return {'level': level, 'four_level_raw_counts': all_counts, 'metrics': data, 'cells': cell_summaries,
            'global': {key: float(global_row[key]) for key in METRICS}}


def summarize_routed(routed, mask=None, weights=None):
    n = len(routed['level'])
    mask = np.ones(n, bool) if mask is None else np.asarray(mask)
    need(mask.dtype == bool and mask.shape == (n,), 'Aggregate row mask')
    levels = {str(j): int(np.count_nonzero(mask & (routed['level'] == j))) for j in range(5)}
    out = {'rows': int(mask.sum()), 'selected_level_rows': levels,
           'four_level_raw_counts': {str(j): describe(routed['four_level_raw_counts'][mask, j]) for j in range(4)},
           'selected_support_and_exposure': {key: describe(value[mask]) for key, value in routed['metrics'].items()}}
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        need(w.shape == (n,), 'Aligned saved weight mass')
        out['saved_date_weight_mass'] = float(w[mask].sum())
        out['selected_level_date_weight_mass'] = {str(j): float(w[mask & (routed['level'] == j)].sum()) for j in range(5)}
    return out


def fit_policy(fit_keys, groups, weights, folds):
    n = len(groups)
    need(folds.dtype == np.int64 and folds.shape == (n,) and set(folds).issubset(set(range(5))), 'Five-fold fit policy')
    output = {'level': np.full(n, -1, np.int8), 'four_level_raw_counts': np.zeros((n, 4), np.int64),
              'metrics': {key: np.zeros(n, np.float64) for key in (*METRICS, 'same_cluster_peers_reference',
                   'same_cluster_peers_selected_cell', 'same_cluster_selected_row_share')}}
    summaries = []
    for fold in range(5):
        held, reference = folds == fold, folds != fold
        need(reference.any(), 'Empty fold complement')
        if held.any():
            r = route(fit_keys, groups, weights, reference, [k[held] for k in fit_keys], groups[held])
            output['level'][held] = r['level']
            output['four_level_raw_counts'][held] = r['four_level_raw_counts']
            for key in output['metrics']:
                output['metrics'][key][held] = r['metrics'][key]
            summaries.append({'fold': fold, 'held_rows': int(held.sum()), 'reference_rows': int(reference.sum()),
                              'reference_cell_summary': r['cells'], 'reference_global': r['global']})
        else:
            summaries.append({'fold': fold, 'held_rows': 0, 'reference_rows': int(reference.sum()),
                              'reference_cell_summary': None, 'reference_global': None})
    need((output['level'] >= 0).all(), 'Every fit row routed exactly once')
    return output, summaries


def inspect_origin(X, Xq, fi, qi, saved_folds, saved_weights, cutoff):
    """Aggregate-only result; caller validates provenance and outer dates."""
    check_X(X); check_X(Xq); check_weights(saved_weights, len(X))
    before = (frame_identity(X), frame_identity(Xq), identity(fi), identity(qi), identity(saved_folds), identity(saved_weights))
    groups, missing = clusters(X, fi)
    qgroups, qmissing = clusters(Xq, qi)
    need(saved_folds.dtype == np.int64 and saved_folds.shape == (len(X),)
         and np.array_equal(saved_folds, original_folds(groups)), 'Saved original group fold replay exact')
    tf = transaction_folds(len(X))
    fk, qk = keys(X), keys(Xq)
    group, group_summary = fit_policy(fk, groups, saved_weights, saved_folds)
    transaction, transaction_summary = fit_policy(fk, groups, saved_weights, tf)
    need((group['metrics']['same_cluster_peers_reference'] == 0).all(), 'Group fold must exclude every same-cluster peer')
    query = route(fk, groups, saved_weights, np.ones(len(X), bool), qk, qgroups)
    labels = query_status_masks(X, Xq)
    need(np.array_equal(labels['missing'], qmissing), 'Original missing-project definition')
    need(sum(int(m.sum()) for m in labels.values()) == len(Xq), 'All query statuses partitioned')
    transitions = {}
    w = saved_weights.astype(np.float64)
    for a in range(5):
        for b in range(5):
            m = (group['level'] == a) & (transaction['level'] == b)
            transitions[f'{a}->{b}'] = {'rows': int(m.sum()), 'saved_date_weight_mass': float(w[m].sum())}
    need(sum(v['rows'] for v in transitions.values()) == len(X), 'All 25 transitions retained')
    qdate = pd.Timestamp('2000-01-01') + pd.to_timedelta(Xq.trade_day_index.to_numpy(dtype=np.float64), unit='D')
    start = pd.Timestamp(cutoff)
    months = np.asarray((qdate.year - start.year) * 12 + (qdate.month - start.month) - (qdate.day < start.day).astype(int))
    need(((months >= 0) & (months < 12)).all(), 'Original forecast-relative month boundary')
    qcity = norm(Xq.address_city).to_numpy()
    result = {'fit_rows': len(X), 'query_rows': len(Xq), 'input_identities': {'fit_indices': identity(fi),
                  'query_indices': identity(qi), 'group_fold_ids': identity(saved_folds), 'date_weights': identity(saved_weights)},
              'transaction_fold_identity': identity(tf), 'transaction_fold_rows': [int((tf == j).sum()) for j in range(5)],
              'original_group_fold_rows': [int((saved_folds == j).sum()) for j in range(5)],
              'group_fit': summarize_routed(group, weights=saved_weights),
              'transaction_fit': summarize_routed(transaction, weights=saved_weights),
              'fit_named_project': {'group': summarize_routed(group, ~missing, saved_weights),
                                    'transaction': summarize_routed(transaction, ~missing, saved_weights)},
              'fit_missing_project_proxy': {'group': summarize_routed(group, missing, saved_weights),
                                            'transaction': summarize_routed(transaction, missing, saved_weights)},
              'fold_reference_summaries': {'group': group_summary, 'transaction': transaction_summary},
              'selected_level_transitions': transitions,
              'raw_support_transaction_over_group': describe(transaction['metrics']['raw_rows'] / group['metrics']['raw_rows']),
              'full_fit_query': summarize_routed(query), 'full_fit_query_cell_summary': query['cells'],
              'full_fit_query_global': query['global'],
              'query_by_project_status': {label: summarize_routed(query, m) for label, m in labels.items()},
              'query_by_city': {city: summarize_routed(query, qcity == city) for city in sorted(set(qcity))},
              'query_by_forecast_quarter': {str(j + 1): summarize_routed(query, months // 3 == j) for j in range(4)},
              'structural_presence_not_model_gate': {
                  'any_selected_level_changes': bool(np.any(group['level'] != transaction['level'])),
                  'any_transaction_same_cluster_exposure': bool(np.any(transaction['metrics']['same_cluster_peers_reference'] > 0)),
                  'any_selected_project_mass_ESS_changes': bool(np.any(group['metrics']['date_weight_project_ESS'] != transaction['metrics']['date_weight_project_ESS']))},
              'same_query_reference_for_both_policies': True, 'automatic_model_experiment_authorized': False,
              'prices_used': False, 'query_labels_read': False, 'raw_rows_are_not_independent_projects': True,
              'project_mass_ESS_is_weight_concentration_only': True, 'Private_score': None, 'Public_score': None}
    after = (frame_identity(X), frame_identity(Xq), identity(fi), identity(qi), identity(saved_folds), identity(saved_weights))
    need(before == after, 'Pure-X input state changed')
    result['inputs_unchanged'] = True
    return result


class TypedReader:
    def __init__(self, spec):
        self.spec, self.opened = spec, set()

    def artifact(self, name):
        permit_artifact(name, self.spec)
        need(sha(name) == self.spec['input_sha256'][name], 'Target-free artifact hash before read')
        self.opened.add(name)
        if safe(name).suffix == '.npy':
            value = np.load(safe(name), allow_pickle=False)
            need(isinstance(value, np.ndarray) and not value.dtype.hasobject, 'Standalone nonobject NPY only')
        else:
            value = pd.read_pickle(safe(name))
            check_X(value)
        need(sha(name) == self.spec['input_sha256'][name], 'Target-free artifact hash after read')
        return value


def day(s):
    return int((pd.Timestamp(s) - pd.Timestamp('2000-01-01')).days)


def outer_dates(X, Xq, start, end):
    for frame in (X, Xq):
        days = frame.trade_day_index.to_numpy(dtype=np.float64)
        need(np.isfinite(days).all() and frame.trade_date_invalid.eq(0).all()
             and frame.trade_date_out_of_range.eq(0).all(), 'Valid original outer-split dates required')
    f, q = X.trade_day_index.to_numpy(), Xq.trade_day_index.to_numpy()
    need((f < day(start)).all() and ((q >= day(start)) & (q <= day(end))).all(), 'Outer date boundary')


def load_origin(stage, spec, reader, raw=None, mapping=None, schema=None):
    origin = spec['origins'][stage]
    paths = origin['paths']
    fi, qi = reader.artifact(paths['fit_indices']), reader.artifact(paths['query_indices'])
    check_index(fi, 455255); check_index(qi, 455255)
    need(not np.intersect1d(fi, qi).size and len(fi) == origin['fit_rows'] and len(qi) == origin['query_rows'],
         'Original fit/query index sizes and nonoverlap')
    need(values_sha(fi) == origin['index_contract']['fit_indices_content_sha256']
         and values_sha(qi) == origin['index_contract']['query_indices_content_sha256'], 'Canonical index value SHA')
    if origin['X_mode'] == 'canonical_from_cache':
        need(raw is not None and mapping is not None and schema is not None, 'Declared pure-X cache context required')
        X, Xq = (raw.iloc[mapping[ix]].reset_index(drop=True).copy(deep=True) for ix in (fi, qi))
        d = raw.trade_day_index.to_numpy(dtype=np.float64)[mapping]
        start, end = WINDOWS[stage][:2]
        expected_f = np.flatnonzero(np.isfinite(d) & (d < day(start))).astype(np.int64)
        expected_q = np.flatnonzero(np.isfinite(d) & (d >= day(start)) & (d <= day(end))).astype(np.int64)
        need(fi.tobytes() == expected_f.tobytes() and qi.tobytes() == expected_q.tobytes(), 'Original full canonical cutoff masks')
    else:
        X, Xq = reader.artifact(paths['Xfit']), reader.artifact(paths['Xquery'])
        source = reader.artifact(paths['nn_source_indices'])
        need(identity(source) == identity(fi), 'Offset NN original canonical source-index identity')
        for name, frame in (('Xfit', X), ('Xquery', Xq)):
            expected = origin['frames'][name]
            actual = frame_identity(frame)
            need(actual['rows'] == expected['rows'] and actual['columns'] == expected['columns']
                 and actual['dtypes'] == expected['dtypes'] and actual['values_sha256'] == expected['frame_sha256'],
                 'Offset original target-free frame identity')
    check_X(X, schema['columns']); check_X(Xq, schema['columns'])
    need(len(X) == len(fi) and len(Xq) == len(qi), 'Feature/index alignment')
    outer_dates(X, Xq, *WINDOWS[stage][:2])
    folds, weights = reader.artifact(paths['group_folds']), reader.artifact(paths['weights'])
    check_weights(weights, len(X))
    need(values_sha(weights) == origin['weights_values_sha256'], 'Original R63/dateweight value SHA')
    return X, Xq, fi, qi, folds, weights


def validate_preflight(preflight, spec):
    need(preflight['status'] == 'passed'
         and preflight['source_sha256'] == {**spec['source_sha256'], PLAN: spec['plan']['sha256']}
         and preflight['input_sha256'] == spec['input_sha256'] and preflight['parameters'] == parameters()
         and preflight['approval_for_pure_X_audit'] is True
         and preflight['real_X_or_arrays_deserialized'] is False
         and preflight['new_fits'] == 0 and preflight['query_labels_read'] is False,
         'Exact independent source/plan/readset pure-X preflight required')


def freeze():
    spec = specification()
    need(not spec['blockers'], 'Missing standalone target-free file; never substitute NPZ or skip origin')
    need(not safe(OUT).exists(), 'Preserve existing diagnostic output/request')
    need(safe(PREFLIGHT).is_file(), 'Independent preflight not yet present')
    preflight = read_json(PREFLIGHT)
    validate_preflight(preflight, spec)
    verify_hashes(spec['input_sha256'])
    verify_hashes({**spec['source_sha256'], PLAN: spec['plan']['sha256']})
    safe(OUT).mkdir(parents=True, exist_ok=False)
    request = {'status': 'pure_X_prior_support_audit_frozen_before_data_deserialization',
               'created_at_utc': now(), 'specification': spec, 'independent_preflight': record(PREFLIGHT),
               'parameters': parameters(), 'source_sha256': spec['source_sha256'], 'plan': spec['plan'],
               'input_sha256': spec['input_sha256'], 'real_X_or_arrays_deserialized': False,
               'input_byte_hashes_verified_at_freeze': True,
               'new_fits': 0, 'GPU_calls': 0, 'query_labels_read': False, 'new_CSV': False}
    write(OUT + '/request.json', request)
    print(json.dumps({'status': 'frozen', 'request': record(OUT + '/request.json'),
                      'input_files': len(spec['input_sha256']), 'new_fits': 0}))


def check_request():
    request = read_json(OUT + '/request.json')
    need(safe(OUT + '/request.sha256').read_text(encoding='ascii').strip() == sha(OUT + '/request.json'), 'Request sidecar')
    spec = specification()
    need(not spec['blockers'] and request['status'] == 'pure_X_prior_support_audit_frozen_before_data_deserialization'
         and request['specification'] == spec and request['parameters'] == parameters()
         and request['source_sha256'] == spec['source_sha256'] and request['plan'] == spec['plan']
         and request['input_sha256'] == spec['input_sha256'] and request['independent_preflight'] == record(PREFLIGHT),
         'Immutable full request/metadata/source/readset identity')
    validate_preflight(read_json(PREFLIGHT), spec)
    verify_hashes({**spec['input_sha256'], **spec['source_sha256'], PLAN: spec['plan']['sha256']})
    return request, spec


def run():
    tick = time.monotonic()
    request, spec = check_request()
    need(not safe(OUT + '/started.json').exists() and not safe(OUT + '/results.json').exists()
         and not safe(OUT + '/failure.json').exists(), 'Existing attempt; no retry or overwrite')
    write(OUT + '/started.json', {'created_at_utc': now(), 'request': record(OUT + '/request.json'),
                                 'real_X_or_arrays_deserialized': False, 'new_fits': 0})
    try:
        schema = read_json(SCHEMA)
        reader = TypedReader(spec)
        raw = reader.artifact(FEATURES)
        check_X(raw, schema['columns'])
        need(len(raw) == 455784, 'Original pure-X raw cache row count')
        raw_before = frame_identity(raw)
        mapping = reader.artifact(MAPPING)
        check_index(mapping, len(raw))
        need(len(mapping) == 455255, 'Original canonical mapping rows')
        mapping_before = identity(mapping)
        records = {}
        for stage in STAGES:
            start = time.monotonic()
            context = load_origin(stage, spec, reader, raw, mapping, schema)
            result = inspect_origin(*context, WINDOWS[stage][0])
            del context
            result.update(round=115, stage=stage, status='pure_X_support_exposure_completed',
                          elapsed_seconds=time.monotonic() - start, cutoff=WINDOWS[stage][0], query_end=WINDOWS[stage][1],
                          request=record(OUT + '/request.json'), source_sha256=spec['source_sha256'], plan=spec['plan'],
                          selected_origin_source=spec['origins'][stage], parameters=parameters())
            safe(OUT + '/' + stage).mkdir(exist_ok=False)
            write(OUT + '/' + stage + '/aggregates.json', result)
            records[stage] = record(OUT + '/' + stage + '/aggregates.json')
            print(json.dumps({'stage': stage, 'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)
        need(frame_identity(raw) == raw_before and identity(mapping) == mapping_before, 'Original pure-X cache/mapping mutated')
        verify_hashes({**spec['input_sha256'], **spec['source_sha256'], PLAN: spec['plan']['sha256']})
        result = {'round': 115, 'status': 'all_four_origins_pure_X_audit_completed', 'created_at_utc': now(),
                  'request': record(OUT + '/request.json'), 'independent_preflight': record(PREFLIGHT),
                  'plan': spec['plan'], 'source_sha256': spec['source_sha256'], 'input_sha256': spec['input_sha256'],
                  'parameters': parameters(), 'stages': records, 'elapsed_seconds': time.monotonic() - tick,
                  'deserialized_artifacts': sorted(reader.opened), 'full_target_free_cache_deserialized': True,
                  'inputs_and_sources_unchanged': True, 'new_fits': 0, 'GPU_calls': 0, 'query_labels_read': False,
                  'raw_price_pickle_opened': False, 'NPZ_opened': False, 'model_loads': 0, 'price_prior_builds': 0,
                  'MAE_calls': 0, 'bootstrap_calls': 0, 'new_CSV': False, 'automatic_model_experiment_authorized': False,
                  'Public_score': None, 'Private_score': None,
                  'interpretation': 'Only support, membership and date-mass concentration; prices and model benefit remain unobserved.'}
        write(OUT + '/results.json', result)
        print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds'], 'new_fits': 0}), flush=True)
    except BaseException as error:
        if not safe(OUT + '/failure.json').exists():
            write(OUT + '/failure.json', {'status': 'failed_preserved_no_retry', 'created_at_utc': now(),
                'error_type': type(error).__name__, 'error_message_omitted_to_avoid_raw_key_disclosure': True,
                'request': record(OUT + '/request.json'),
                'elapsed_seconds': time.monotonic() - tick, 'new_fits': 0, 'GPU_calls': 0, 'new_CSV': False})
        raise


def self_check():
    """Artificial CPU data only; never calls specification, freeze, reader or run."""
    checks = []
    def mark(name, condition):
        need(bool(condition), name)
        checks.append(name)
    def reject(name, call):
        try:
            call()
        except (ValueError, KeyError, TypeError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    tf = transaction_folds(103)
    expected = np.empty(103, np.int64)
    expected[np.random.default_rng(42).permutation(103)] = np.arange(103, dtype=np.int64) % 5
    mark('exact_seed42_permutation_then_balanced_assign', tf.tobytes() == expected.tobytes())
    sizes = np.bincount(tf, minlength=5)
    mark('balanced_fold_sizes_differ_at_most_one', int(sizes.max() - sizes.min()) == 1)
    mark('new_rng_for_each_origin_reproducible', tf.tobytes() == transaction_folds(103).tobytes())
    reject('less_than_five_rows_refused', lambda: transaction_folds(4))
    n = 200
    cutoff = '2023-02-11'
    X = pd.DataFrame({'address_city': ['CityA'] * n, 'address_district': ['DistrictA'] * 160 + ['DistrictB'] * 40,
        'transaction_type': ['house'] * n, 'building_type': ['apartment'] * n,
        'project_name': [f'secret-fixture-project-{i // 20}' for i in range(n)],
        'address_raw': ['__MISSING__'] * n, 'address_road': ['__MISSING__'] * n, 'address_section': ['__MISSING__'] * n,
        'trade_day_index': np.full(n, day(cutoff) - 100, np.float32),
        'trade_date_invalid': np.zeros(n, np.float32), 'trade_date_out_of_range': np.zeros(n, np.float32)})
    fi = np.arange(1000, 1000 + n, dtype=np.int64)
    qi = np.arange(2000, 2004, dtype=np.int64)
    Xq = X.iloc[[0, 1, 2, 3]].reset_index(drop=True).copy()
    Xq.loc[1, 'project_name'] = 'secret-fixture-new-project'
    Xq.loc[2, 'project_name'] = '__MISSING__'
    Xq['trade_day_index'] = np.asarray([day(cutoff), day(cutoff) + 1, day(cutoff) + 120, day(cutoff) + 300], np.float32)
    g, missing = clusters(X, fi)
    gf = original_folds(g)
    mark('whole_project_original_hash_fold', all(len(np.unique(gf[g == name])) == 1 for name in np.unique(g)))
    manual = int.from_bytes(hashlib.sha256(('42|' + json.dumps(['project', 'CityA', 'DistrictA', 'secret-fixture-project-0'],
                            ensure_ascii=False, separators=(',', ':'))).encode('utf-8')).digest()[:8], 'little') % 5
    mark('original_cluster_key_hash_exact', gf[0] == manual and not missing.any())
    weights = np.linspace(.5, 1.5, n, dtype=np.float32)
    r = inspect_origin(X, Xq, fi, qi, gf, weights, cutoff)
    mark('all25_transitions_retained', len(r['selected_level_transitions']) == 25
         and sum(v['rows'] for v in r['selected_level_transitions'].values()) == n)
    mark('original_group_peer_exposure_zero', r['group_fit']['selected_support_and_exposure']['same_cluster_peers_reference']['max'] == 0)
    mark('transaction_peer_exposure_present', r['transaction_fit']['selected_support_and_exposure']['same_cluster_peers_reference']['max'] > 0)
    mark('query_seen_new_missing_partition', [r['query_by_project_status'][k]['rows'] for k in ('seen', 'new_nonmissing', 'missing')] == [2, 1, 1])
    legacy_fit = pd.DataFrame({'address_city': ['City|Part'], 'address_district': ['District'], 'project_name': ['Project']})
    legacy_query = pd.DataFrame({'address_city': ['City'], 'address_district': ['Part|District'], 'project_name': ['Project']})
    mark('legacy_pipe_membership_preserved_exact', query_status_masks(legacy_fit, legacy_query)['seen'].tolist() == [True])
    mark('all_forecast_quarters_include_empty', len(r['query_by_forecast_quarter']) == 4
         and sum(x['rows'] for x in r['query_by_forecast_quarter'].values()) == 4)
    mark('aggregate_output_no_project_identifiers', 'secret-fixture-' not in json.dumps(r, ensure_ascii=False))
    mark('model_not_authorized_by_structural_presence', r['automatic_model_experiment_authorized'] is False)
    mark('query_input_does_not_change_fit_policy', r['group_fit'] == inspect_origin(X, Xq.iloc[::-1].reset_index(drop=True), fi,
         qi[::-1].copy(), gf, weights, cutoff)['group_fit'])
    table, _ = group_table(np.zeros(30, int), np.array(['single'] * 30), np.ones(30, np.float32))
    mark('30_rows_one_project_ESS_one', table.loc[0, 'raw_rows'] == 30 and table.loc[0, 'project_groups'] == 1
         and table.loc[0, 'date_weight_project_ESS'] == 1)
    table, _ = group_table(np.zeros(3, int), np.array(['a', 'a', 'b']), np.array([1., 1., 2.], np.float32))
    mark('weighted_project_mass_ESS_formula', table.loc[0, 'date_weight_project_ESS'] == 2
         and table.loc[0, 'largest_project_date_mass_share'] == .5 and table.loc[0, 'largest_project_row_share'] == 2 / 3)
    small = X.iloc[:61].copy().reset_index(drop=True)
    small['address_district'] = ['A'] * 30 + ['B'] * 29 + ['C'] * 2
    sk = keys(small)
    sg, _ = clusters(small, np.arange(61, dtype=np.int64))
    q = small.iloc[[0, 30, 60]].reset_index(drop=True).copy()
    q.loc[2, 'transaction_type'] = 'unseen_type'
    qg, _ = clusters(q, np.arange(500, 503, dtype=np.int64))
    routed = route(sk, sg, np.ones(61, np.float32), np.ones(61, bool), keys(q), qg)
    mark('raw30_supported_raw29_fallback', routed['level'].tolist() == [0, 1, 4])
    routed29 = route([k[:29] for k in sk], sg[:29], np.ones(29, np.float32), np.ones(29, bool), keys(q), qg)
    mark('global_has_no30_row_minimum', (routed29['level'] == 4).all() and (routed29['metrics']['raw_rows'] == 29).all())
    mark('global_and_all_levels_positive_finite', all(np.isfinite(v).all() for v in routed29['metrics'].values()))
    miss = X.iloc[:4].copy().reset_index(drop=True)
    miss['project_name'] = '__MISSING__'
    miss.loc[0, 'address_raw'] = 'address-first'
    miss.loc[0, 'address_road'] = 'road-also'
    miss.loc[1, 'address_road'] = 'road-only'
    mg, mm = clusters(miss, np.arange(700, 704, dtype=np.int64))
    mark('missing_address_road_canonical_row_fallbacks', mm.all() and json.loads(mg[0])[0] == 'missing_address'
         and json.loads(mg[1])[0] == 'missing_road' and json.loads(mg[2]) == ['missing_row', 702])
    mark('normalization_original_missing_policy', norm(pd.Series([None, '', 'nan', 'NaT', ' x '])).tolist()
         == ['__MISSING__'] * 4 + ['x'])
    reject('target_column_rejected', lambda: check_X(X.assign(query_y=1.)))
    reject('weight_float64_rejected', lambda: check_weights(weights.astype(np.float64), n))
    reject('weight_zero_rejected', lambda: check_weights(np.zeros(n, np.float32), n))
    reject('weight_nonfinite_rejected', lambda: check_weights(np.full(n, np.nan, np.float32), n))
    badfold = gf.copy(); badfold[0] = (badfold[0] + 1) % 5
    reject('wrong_saved_group_fold_rejected', lambda: inspect_origin(X, Xq, fi, qi, badfold, weights, cutoff))
    reject('overlapping_or_unsorted_index_rejected', lambda: check_index(np.array([2, 2], np.int64), 4))
    reject('wrong_outer_cutoff_rejected', lambda: outer_dates(Xq, X, cutoff, '2024-02-10'))
    fake = {'origins': {'x': {'paths': {'weights': 'forbidden.npz'}}}, 'input_sha256': {'forbidden.npz': 'x'}}
    reject('NPZ_rejected_before_any_array_read', lambda: permit_artifact('forbidden.npz', fake))
    reject('raw_price_pickle_outside_readset', lambda: permit_artifact('data_cache/train.pkl', fake))
    reject('reference_price_pickle_outside_readset', lambda: permit_artifact('outputs/reference.pkl', fake))
    reject('outside_workspace_rejected', lambda: safe('../outside.npy'))
    spec = {'source_sha256': {'src/example.py': 'x'}, 'plan': {'sha256': 'p'}, 'input_sha256': {'a.npy': 'a'}}
    pf = {'status': 'passed', 'source_sha256': {'src/example.py': 'x', PLAN: 'p'}, 'input_sha256': {'a.npy': 'a'},
          'parameters': parameters(), 'approval_for_pure_X_audit': True, 'real_X_or_arrays_deserialized': False,
          'new_fits': 0, 'query_labels_read': False}
    validate_preflight(pf, spec)
    mark('exact_independent_preflight_contract', True)
    for key, bad in [('input_sha256', {}), ('source_sha256', {}), ('approval_for_pure_X_audit', False),
                     ('real_X_or_arrays_deserialized', True), ('query_labels_read', True), ('new_fits', 1)]:
        reject('preflight_' + key + '_mismatch_rejected', lambda key=key, bad=bad: validate_preflight({**pf, key: bad}, spec))
    mark('synthetic_inputs_unmutated', r['inputs_unchanged'])
    return {'status': 'passed', 'check_count': len(checks), 'checks': checks, 'source_sha256': sha(str(Path(__file__).relative_to(ROOT))),
            'real_X_or_arrays_deserialized': False, 'prices_read': False, 'query_labels_read': False,
            'new_fits': 0, 'GPU_calls': 0, 'new_CSV': False, 'prepared_source_not_execution_authorization': True,
            'author_selfcheck_not_independent_preflight': True}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check', action='store_true')
    mode.add_argument('--describe', action='store_true')
    mode.add_argument('--freeze', action='store_true')
    mode.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    elif args.describe:
        print(json.dumps(specification(), ensure_ascii=False, sort_keys=True, allow_nan=False))
    elif args.freeze:
        freeze()
    else:
        run()
