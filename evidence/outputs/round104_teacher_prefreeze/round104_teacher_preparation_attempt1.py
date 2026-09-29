"""R104 early teachers: separate pure-X preparation and fit-label extraction.

The 2023 non-price geography is deliberately static, NOT historically as-of
2020. R90's historical-availability guard is unchanged and never bypassed by
calling its high-level loader. Only its pure frame/index validators are used.
"""
from __future__ import annotations
import argparse
import json
import time
import traceback
import numpy as np
import pandas as pd
import round100_contract as c
import round90_offset_data as pure
import round6_geo as geo
import round47_nn_encode as enc
import round54_numeric_encode as ramp
import train_lightgbm as light

PLAN = 'reports/round104_plan.md'
PLAN_SHA = '106fda0c56cc7c5660c6255bb572fd78c59ccac9c13d25604bbe592204e43778'
ORIGIN, QUERY_END = '2020-08-11', '2021-08-10'
P79 = 'outputs/round79/plan.json'
P79_SHA = '7b46ade28f061ddf989fd50692f154e02b3bf73facf5a4cb408d415d09d1ea1f'
GEO_MANIFEST = 'outputs/round6/geography/confirmation/manifest.json'
CONFIG = 'runs/v1/models/final_config.json'
CONFIG_SHA = '2ca4258f363d1937df50aed72f5b11dc32b7e70d9dd5fb57fb6f375673a2a425'
TRAINER = 'src/round48_nn_train.py'
TRAINER_SHA = '38fb91e01eabb2ed3163f1260e16c826ca55592951c22ce1107ad68c24b37082'
ARCHITECTURE = 'src/round47_nn_train.py'
ARCHITECTURE_SHA = '68175c331e3fe31bdc298a9b37035fcbf51b78424749454dfde4a29e9cd65bb7'
INDEX_NAMES = ('fit_indices', 'query_indices', 'fit_source_cache_indices', 'query_source_cache_indices')
PURE_ARRAYS = ('train_num', 'train_cat', 'query_num', 'query_cat', 'cardinalities',
               'train_area_factor', 'query_area_factor')
GEO_LIMIT = {'snapshot_year': 2023, 'replication_timestamp': '2023-01-01T21:21:53Z',
             'geographic_values_learned_from_targets': False, 'fully_historical_as_of': False,
             'future_road_availability_bias_acknowledged': True,
             'R90_original_guard_unchanged': True}


def source_hashes():
    return c.source_hashes('src/round104_teacher_preparation.py',
                          'src/round104_teacher_fit.py', TRAINER)


def parameters():
    return {'stage': 'pre_auxiliary', 'origin': ORIGIN, 'query_end': QUERY_END,
            'fit_rule': 'finite trade_day_index strictly before origin',
            'query_rule': 'finite trade_day_index between origin and query_end inclusive',
            'area_NN': 'immutable R48 trainer with new fit-only R47 encoder and R54 ramps',
            'LGB': 'original v1 unit 3999 trees, X87 only', 'geography': GEO_LIMIT,
            'test_read': False, 'query_labels_exported': False}


def partition(days):
    days = np.asarray(days, dtype=np.float64)
    first = (pd.Timestamp(ORIGIN) - pd.Timestamp('2000-01-01')).days
    last = (pd.Timestamp(QUERY_END) - pd.Timestamp('2000-01-01')).days
    return (np.flatnonzero(np.isfinite(days) & (days < first)).astype(np.int64),
            np.flatnonzero(np.isfinite(days) & (days >= first) & (days <= last)).astype(np.int64))


def check_record(rec):
    c.need(c.sha(rec['path']) == rec['sha256'], 'Record SHA differs: ' + rec['path'])
    return c.safe(rec['path'])


def authority():
    """Metadata and file hashes only; never deserialize arrays or price tables."""
    inputs = {}
    def bind(name, digest):
        c.need(c.sha(name) == digest, 'Authority SHA differs: ' + name)
        inputs[name] = digest
    bind(PLAN, PLAN_SHA); bind(P79, P79_SHA); bind(CONFIG, CONFIG_SHA)
    p79 = c.read(P79)
    bind(GEO_MANIFEST, p79['input_sha256'][GEO_MANIFEST]); m = c.read(GEO_MANIFEST)
    for name in (c.FEATURES, c.CANONICAL, pure.SCHEMA, c.RAW):
        bind(name, p79['input_sha256'][name])
    c.need(m['source_sha256'][c.FEATURES] == inputs[c.FEATURES] and
           m['source_sha256'][c.RAW] == inputs[c.RAW], 'Raw/cache geographic provenance differs')
    c.need(m['index_files_sha256']['outputs/round6/geography/confirmation/train_source_cache_indices.npy'] ==
           inputs[c.CANONICAL], 'Canonical geographic mapping differs')
    c.need(m['alignment']['train']['canonical_rows'] == 455255 and
           m['alignment']['train']['header_removed_cache_rows'] == 455784 and
           m['alignment']['train']['original_id_address_district_transaction_type_verified'], 'Geo row alignment differs')
    c.need(m['geographic_values_learned_from_targets'] is False and m['price_model_trained'] is False,
           'Geography must not learn targets')
    chain = m['geography_provenance_chain']
    c.need(chain['verified'] and chain['price_data_deserialized'] is False and
           chain['recorded_source_input_output_hashes_verified'] and
           chain['matches_to_index_to_extraction_PBF_verified'], 'Inherited non-price provenance missing')
    c.need(m['snapshot_year'] == 2023 and m['replication_timestamp'] == GEO_LIMIT['replication_timestamp'],
           'Static geographic snapshot differs')
    bind(m['features']['train']['path'], m['features']['train']['sha256'])
    for key in ('geo_index_audit', 'matching_audit'):
        bind(m[key + '_path'], m[key + '_sha256'])
    bind(pure.EXTRACTION, m['source_sha256'][pure.EXTRACTION])
    match = c.read(m['matching_audit_path'])
    c.need(match['target_values_read'] is False and match['model_fitting_or_scoring_performed'] is False,
           'Matching audit target boundary differs')
    extraction = c.read(pure.EXTRACTION)
    c.need(extraction['source']['header']['replication_timestamp'] == m['replication_timestamp'] and
           extraction['source']['sha256'] == m['source_pbf_sha256'], 'Extraction identity differs')
    schema = c.read(pure.SCHEMA)
    c.need(len(schema['columns']) == 87 and len(schema['categorical_columns']) == 21, 'Original schema differs')
    sources = source_hashes()
    for name, digest in sources.items():
        if name in p79['source_sha256']:
            c.need(digest == p79['source_sha256'][name], 'Historical source drift: ' + name)
    c.need(sources[TRAINER] == TRAINER_SHA and sources[ARCHITECTURE] == ARCHITECTURE_SHA, 'Area trainer drift')
    return inputs, sources, m


def freeze(path):
    inputs, sources, m = authority()
    c.safe(path).parent.mkdir(parents=True, exist_ok=True)
    c.write(path, {'schema_version': 1, 'status': 'teacher_preparation_request_frozen',
                   'created_at_utc': c.now(), 'plan_sha256': PLAN_SHA, 'parameters': parameters(),
                   'input_sha256': inputs, 'source_sha256': sources, 'protected_before': c.protected_hashes(),
                   'geography': GEO_LIMIT, 'geo_train': m['features']['train'],
                   'arrays_deserialized': False, 'prices_deserialized': False})


def verify_request(path):
    r = c.read(path)
    c.need(r['status'] == 'teacher_preparation_request_frozen' and r['plan_sha256'] == PLAN_SHA and
           r['parameters'] == parameters() and r['source_sha256'] == source_hashes(), 'Preparation request differs')
    c.verify_hashes(r['source_sha256']); c.verify_hashes(r['input_sha256'])
    c.need(c.protected_hashes() == r['protected_before'], 'Protected files changed')
    return r


def frame_record(path, frame):
    return {**c.record(path), 'rows': len(frame), 'columns': list(frame),
            'dtypes': {k: str(frame[k].dtype) for k in frame}, 'values_sha256': pure.frame_sha(frame)}


def save_frame(path, frame):
    c.need(not c.safe(path).exists(), 'Frame exists')
    pure.check_target_free(frame, 'saved features'); frame.to_pickle(c.safe(path))
    pd.testing.assert_frame_equal(frame, pd.read_pickle(c.safe(path)), check_exact=True)
    return frame_record(path, frame)


def load_frame(rec):
    p = check_record(rec); f = pd.read_pickle(p); pure.check_target_free(f, 'loaded features')
    c.need(frame_record(p, f) == rec and f.index.equals(pd.RangeIndex(len(f))), 'Frame identity differs')
    return f


def save_npz(path, arrays):
    with c.safe(path).open('xb') as f:
        np.savez(f, **arrays)
    with np.load(c.safe(path), allow_pickle=False) as z:
        c.need(set(z.files) == set(arrays), 'NPZ keys differ')
        for key, a in arrays.items(): c.exact(a, z[key], 'NPZ reload differs: ' + key)
    return {**c.record(path), 'arrays': {k: c.identity(v) for k, v in arrays.items()}}


def prepare(request, out):
    r = verify_request(request); out = c.safe(out); out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic(); status = 'failed'; error = None
    try:
        raw = pd.read_pickle(c.safe(c.FEATURES)); mapping = np.load(c.safe(c.CANONICAL), allow_pickle=False)
        schema = c.read(pure.SCHEMA)
        X = pure.canonical_features(raw, mapping, schema['columns']); del raw
        c.need(len(X) == 455255 and len(mapping) == 455255, 'Canonical rows differ')
        G = pd.read_pickle(c.safe(r['geo_train']['path'])); pure.check_geo(G)
        c.need(len(G) == len(X) and G.index.equals(X.index), 'Geo canonical alignment differs')
        fi, qi = partition(X.trade_day_index.to_numpy())
        pure.check_index(fi, len(X), 'fit'); pure.check_index(qi, len(X), 'query')
        c.need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
        indices = dict(zip(INDEX_NAMES, (fi, qi, mapping[fi], mapping[qi])))
        index_records = {k: c.save_array(out / (k + '.npy'), v) for k, v in indices.items()}
        frames = {'Xfit': X.iloc[fi].reset_index(drop=True), 'Xquery': X.iloc[qi].reset_index(drop=True),
                  'Gfit': G.iloc[fi].reset_index(drop=True), 'Gquery': G.iloc[qi].reset_index(drop=True)}
        del X, G
        frame_records = {k: save_frame(out / (k + '.pkl'), v) for k, v in frames.items()}
        aug = {s: geo.augment_features(frames['X' + s], frames['G' + s]) for s in ('fit', 'query')}
        cats = schema['categorical_columns']; state = enc.fit_encoder(aug['fit'], geo.categories(frames['Xfit'], cats))
        tn, tc = enc.encode(state, aug['fit']); qn, qc = enc.encode(state, aug['query'])
        rs = ramp.fit_state(tn)
        c.write(out / 'encoder_state.json', state); c.write(out / 'ramp_state.json', rs)
        restored = c.read(out / 'encoder_state.json'); restored_ramp = c.read(out / 'ramp_state.json')
        for role, expected in (('fit', (tn, tc)), ('query', (qn, qc))):
            observed = enc.encode(restored, aug[role])
            for a, b in zip(expected, observed): c.exact(a, b, 'Encoder JSON reload differs')
        arrays = {'train_num': ramp.transform(rs, tn), 'train_cat': tc,
                  'query_num': ramp.transform(rs, qn), 'query_cat': qc,
                  'cardinalities': np.asarray(state['cardinalities'], dtype=np.int64),
                  'train_area_factor': (light.effective_area(frames['Xfit']) / 100).astype(np.float32),
                  'query_area_factor': (light.effective_area(frames['Xquery']) / 100).astype(np.float32)}
        for name, a in (('train_num', tn), ('query_num', qn)):
            c.exact(arrays[name], ramp.transform(restored_ramp, a), 'Ramp JSON reload differs')
        for key in ('train_area_factor', 'query_area_factor'):
            c.need(np.isfinite(arrays[key]).all() and (arrays[key] > 0).all(), 'Invalid pure-X area factor')
        encoded, maps, names, catcols = light.encode_features({'train': frames['Xfit'], 'query': frames['Xquery']}, cats)
        category = {'category_maps': maps, 'safe_names': names, 'categorical_columns': catcols,
                    'original_categorical_columns': cats, 'fit_rows': len(fi), 'fit_uses_query': False, 'fit_uses_y': False}
        c.write(out / 'lgb_category_state.json', category)
        restored_maps = c.read(out / 'lgb_category_state.json')['category_maps']
        encoded2, maps2, names2, cats2 = light.encode_features({'train': frames['Xfit'], 'query': frames['Xquery']}, cats, restored_maps)
        c.need(maps2 == maps and names2 == names and cats2 == catcols, 'LGB category JSON reload differs')
        for role in encoded: pd.testing.assert_frame_equal(encoded[role], encoded2[role], check_exact=True)
        prepared = save_npz(out / 'area_pure_X.npz', arrays)
        states = {k: c.record(out / (k + '.json')) for k in ('encoder_state', 'ramp_state', 'lgb_category_state')}
        m = {'schema_version': 1, 'status': 'pure_X_teacher_prepared', 'stage': 'pre_auxiliary',
             'origin': ORIGIN, 'cutoff': ORIGIN, 'query_end': QUERY_END, 'fit_rows': len(fi), 'query_rows': len(qi),
             'plan_sha256': PLAN_SHA, 'request': c.record(request), 'source_sha256': r['source_sha256'],
             'input_sha256': r['input_sha256'], 'features': c.record(c.FEATURES), 'canonical_mapping': c.record(c.CANONICAL),
             'raw_train': {'path': c.RAW, 'sha256': r['input_sha256'][c.RAW], 'rows': 455784, 'declared_only': True},
             'indices': index_records, 'frames': frame_records, 'states': states, 'prepared': prepared,
             'geography': GEO_LIMIT, 'categorical_columns': cats, 'encoder_and_ramp_fit_only': True,
             'query_labels_available': False, 'raw_price_pickle_opened': False, 'new_native_fits': 0,
             'protected_before': r['protected_before'], 'protected_after': c.protected_hashes(),
             'elapsed_seconds': time.monotonic() - t}
        c.need(m['protected_before'] == m['protected_after'], 'Protected files changed')
        c.write(out / 'manifest.json', m); status = 'completed'
        return m
    except Exception:
        error = traceback.format_exc(); raise
    finally:
        c.write(out / 'execution.json', {'status': status, 'exit_code': 0 if status == 'completed' else 1,
                'failure': error, 'elapsed_seconds': time.monotonic() - t, 'new_native_fits': 0})


def load_prepared(path):
    m = c.read(path)
    c.need(m['status'] == 'pure_X_teacher_prepared' and m['plan_sha256'] == PLAN_SHA and
           m['origin'] == ORIGIN and m['query_end'] == QUERY_END and m['geography'] == GEO_LIMIT, 'Preparation identity differs')
    c.verify_hashes(m['source_sha256'])
    indices = {k: c.load_array(m['indices'][k]) for k in INDEX_NAMES}
    frames = {k: load_frame(v) for k, v in m['frames'].items()}
    for key, expected in (('Xfit', m['fit_rows']), ('Xquery', m['query_rows'])):
        c.need(len(frames[key]) == expected, 'Frame length differs')
    c.need(len(partition(frames['Xfit'].trade_day_index)[0]) == m['fit_rows'] and
           len(partition(frames['Xquery'].trade_day_index)[1]) == m['query_rows'], 'Prepared dates differ')
    for rec in m['states'].values(): check_record(rec)
    check_record(m['prepared'])
    with np.load(c.safe(m['prepared']['path']), allow_pickle=False) as z:
        c.need(set(z.files) == set(PURE_ARRAYS), 'Expected seven pure-X arrays, no labels')
        arrays = {k: z[k] for k in PURE_ARRAYS}
    c.need({k: c.identity(a) for k, a in arrays.items()} == m['prepared']['arrays'], 'Prepared semantic identity differs')
    return m, indices, frames, arrays


def extract_labels(request, preparation, out):
    r = verify_request(request); m = c.read(preparation)
    c.need(m['request'] == c.record(request) and m['status'] == 'pure_X_teacher_prepared', 'Preparation/request differs')
    out = c.safe(out); out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic(); status = 'failed'; error = None
    try:
        fi = c.load_array(m['indices']['fit_indices']); ri = c.load_array(m['indices']['fit_source_cache_indices'])
        mapping = np.load(c.safe(c.CANONICAL), allow_pickle=False)
        c.exact(mapping[fi], ri, 'Fit canonical/raw mapping differs')
        raw = pd.read_pickle(c.safe(c.RAW))  # Whole table deserialized; only fit rows exported.
        c.need(len(raw) == 455784 and raw.index.equals(pd.RangeIndex(len(raw))), 'Raw cache order differs')
        y = raw.iloc[ri]['總價元'].to_numpy(dtype=np.float64, copy=True); del raw
        c.need(y.shape == (len(fi),) and np.isfinite(y).all() and (y > 0).all(), 'Invalid fit prices')
        yr = c.save_array(out / 'fit_y.npy', y)
        result = {'schema_version': 1, 'status': 'teacher_fit_labels_sealed', 'plan_sha256': PLAN_SHA,
                  'preparation': c.record(preparation), 'request': c.record(request), 'fit_rows': len(fi),
                  'fit_y': yr, 'fit_indices': m['indices']['fit_indices'], 'fit_source_cache_indices': m['indices']['fit_source_cache_indices'],
                  'raw_train': m['raw_train'], 'canonical_mapping': m['canonical_mapping'], 'source_sha256': source_hashes(),
                  'whole_price_pickle_deserialized': True, 'query_prices_exported_or_scored': False,
                  'raw_query_bytes_in_deserialized_table': True, 'protected_before': r['protected_before'],
                  'protected_after': c.protected_hashes(), 'elapsed_seconds': time.monotonic() - t}
        c.need(result['protected_before'] == result['protected_after'], 'Protected files changed')
        c.write(out / 'manifest.json', result); status = 'completed'; return result
    except Exception:
        error = traceback.format_exc(); raise
    finally:
        c.write(out / 'execution.json', {'status': status, 'exit_code': 0 if status == 'completed' else 1,
                'failure': error, 'elapsed_seconds': time.monotonic() - t, 'new_native_fits': 0})


def self_check():
    checks = []
    def check(v, name): c.need(v, name); checks.append(name)
    day = lambda s: (pd.Timestamp(s) - pd.Timestamp('2000-01-01')).days
    fi, qi = partition([np.nan, day('2020-08-10'), day(ORIGIN), day(QUERY_END), day('2021-08-11'), np.inf])
    check(fi.tolist() == [1] and qi.tolist() == [2, 3], 'strict fit and inclusive query date boundaries')
    raw = pd.DataFrame({'trade_day_index': [1., 2., 3., 4.], 'n': [11., 12., 13., 14.]})
    mapping = np.array([0, 2, 3], dtype=np.int64); X = pure.canonical_features(raw, mapping)
    check(X.n.tolist() == [11., 13., 14.] and mapping[np.array([1])].tolist() == [2], 'canonical versus raw index axes')
    a = pd.DataFrame({'building_area': [100., np.nan, -1.], 'parking_area': [3., 20., np.nan]})
    check(light.effective_area(a).tolist() == [100., 20., 1.], 'original area fallback')
    cats = ['c' + str(i) for i in range(23)]
    f = pd.DataFrame({**{'n' + str(i): np.arange(8, dtype=np.float64) for i in range(72)},
                      **{k: ['known'] * 8 for k in cats}})
    q = f.iloc[[1, 4]].reset_index(drop=True).copy(); q[cats[0]] = 'unknown'
    state = enc.fit_encoder(f, cats); tnum, tcat = enc.encode(state, f); qnum, qcat = enc.encode(state, q)
    check(qcat[:, 0].tolist() == [0, 0] and (tcat[:, 0] == 1).all(), 'query category never enters fit vocabulary')
    restored = json.loads(json.dumps(state)); rn, rc = enc.encode(restored, q)
    check(rn.tobytes() == qnum.tobytes() and rc.tobytes() == qcat.tobytes(), 'encoder state JSON reload exact')
    rs = ramp.fit_state(tnum); out = ramp.transform(rs, qnum)
    check(out[:, :144].tobytes() == qnum.tobytes(), 'ramp preserves original numeric prefix')
    check(ramp.transform(json.loads(json.dumps(rs)), qnum).tobytes() == out.tobytes(), 'ramp JSON reload exact')
    le, maps, names, cc = light.encode_features({'train': f, 'query': q}, cats)
    le2, _, _, _ = light.encode_features({'train': f, 'query': q}, cats, json.loads(json.dumps(maps)))
    check(le['query'][names[cats[0]]].tolist() == [-1, -1] and le['query'].equals(le2['query']), 'LGB maps fit only and reload exact')
    check(len(PURE_ARRAYS) == 7 and 'train_y_million' not in PURE_ARRAYS, 'pure-X NPZ has no prices')
    check(GEO_LIMIT['fully_historical_as_of'] is False, 'static geography caveat explicit')
    return {'status': 'passed', 'check_count': len(checks), 'checks': checks, 'real_data_loaded': False,
            'new_native_fits': 0, 'plan_sha256': PLAN_SHA, 'source_sha256': source_hashes()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze-request'); ap.add_argument('--prepare', action='store_true')
    ap.add_argument('--extract-fit-labels', action='store_true'); ap.add_argument('--request')
    ap.add_argument('--preparation'); ap.add_argument('--out'); ap.add_argument('--self-check', action='store_true'); ap.add_argument('--receipt')
    a = ap.parse_args()
    if a.self_check:
        started = time.monotonic(); result = self_check(); result['elapsed_seconds'] = time.monotonic() - started
        c.write(a.receipt, result); print(json.dumps(result, ensure_ascii=False))
    elif a.freeze_request: freeze(a.freeze_request)
    elif a.prepare: prepare(a.request, a.out)
    elif a.extract_fit_labels: extract_labels(a.request, a.preparation, a.out)
    else: ap.error('Choose freeze, prepare, extract-fit-labels or self-check')
