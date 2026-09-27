"""One fixed artificial GPU comparison; execute only after authorization via --run.

No property features, targets, predictions, or models are loaded. The candidate
adds one FloatTargetMeanValue simple CTR. Two trees per model, no search/score.
An exclusive run claim and append-only fit events prevent an unnoticed retry.
"""
from __future__ import annotations
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import time
import traceback

import catboost
from catboost import CatBoostRegressor, Pool
import numpy as np
import pandas as pd
import round6_geo as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/float_ctr_research'
BASELINE = 'outputs/round23_research/baseline_categorical_config.json'
BASELINE_SHA = '68d062a1c15b2bcd1215fe31563788a22025f665356fdc059d65f310e422d93e'
REPORT = 'reports/catboost_float_ctr_research.md'
REPORT_SHA = '5078df0799e39455793d17cc0e2d1d2f45ec30e66a552a74bb7099730e4dac97'
CONFIG = 'runs/v1/models/final_config.json'
FIT_ROWS, QUERY_ROWS, DATA_SEED = 512, 32, 20260926
NAMES = ('mae_control', 'float_ctr_candidate')
EXTRA_CTR = 'FloatTargetMeanValue:CtrBorderCount=15:CtrBorderType=Uniform:Prior=0/1'
SOURCES = ('src/float_ctr_feasibility.py', 'src/round6_geo.py', 'src/prepare_data.py')
OFFICIAL_SOURCES = [
    'https://catboost.ai/docs/en/references/training-parameters/ctr',
    'https://github.com/catboost/catboost/blob/v1.2.10/catboost/cuda/ctrs/ctr_calcers.h#L155-L178',
    'https://github.com/catboost/catboost/blob/v1.2.10/catboost/libs/model/model_export/json_model_helpers.cpp#L95-L106',
    'https://github.com/catboost/catboost/blob/v1.2.10/catboost/libs/model/model_export/json_model_helpers.cpp#L273-L306',
    'https://github.com/catboost/catboost/blob/v1.2.10/catboost/libs/model/model.cpp#L432-L496',
]


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def vector_sha(a):
    return hashlib.sha256(np.asarray(a, dtype='<f8').tobytes()).hexdigest()


def write_json(path, value):
    data = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    with path.with_suffix('.sha256').open('x', encoding='ascii') as f:
        f.write(hashlib.sha256(data).hexdigest() + '\n')


def event(value):
    value = dict(value, recorded_at_utc=datetime.now(timezone.utc).isoformat())
    with (OUT / 'events.jsonl').open('a', encoding='utf-8') as f:
        f.write(json.dumps(value, ensure_ascii=False, allow_nan=False) + '\n')
        f.flush()
        os.fsync(f.fileno())
    print(json.dumps(value, ensure_ascii=False), flush=True)


def artificial_frames():
    rng = np.random.default_rng(DATA_SEED)
    n = FIT_ROWS + QUERY_ROWS
    group = np.arange(n) % 16
    cats = ['category_' + str(i) for i in range(21)]
    X = pd.DataFrame({cats[0]: pd.Series(['g' + str(x) for x in group], dtype=object)})
    for c in cats[1:]:
        X[c] = pd.Series(['g' + str(x) for x in rng.integers(0, 16, size=n)], dtype=object)
    for i in range(66):
        X['numeric_' + str(i)] = rng.normal(size=n).astype('float32')
    geo = pd.DataFrame({c: np.zeros(n, dtype='float32') for c in base.NUMERIC})
    geo['geo_latitude'] = np.full(n, 25., dtype='float32')
    geo['geo_longitude'] = np.full(n, 121., dtype='float32')
    geo['geo_has_position'] = np.ones(n, dtype='float32')
    geo['geo_match_status'] = pd.Series(['road_level'] * n, dtype=object)
    geo['geo_grid_002'] = pd.Series(['1250:6050'] * n, dtype=object)
    view = base.augment_features(X, geo[base.FEATURES])
    categories = base.categories(X, cats)
    # Only fit targets exist. Repeated categories have distinct continuous means
    # even within the same median-target half; other numeric columns are noise.
    i = np.arange(FIT_ROWS)
    y = 1. + (i % 16) / 4. + .01 * (i // 16)
    need(view.shape == (544, 95) and len(categories) == 23, 'Artificial schema must be 95/23')
    need(X[cats[0]].iloc[:FIT_ROWS].nunique() == 16, 'Primary category must exceed one-hot threshold 8')
    return view, np.asarray(y, dtype='<f8'), categories


def ctr_evidence(native):
    result = {'global_ctr_target_border_count': native.get('ctr_target_border_count')}
    for family in ('simple_ctr', 'combinations_ctr'):
        need(isinstance(native.get(family), list), 'Missing native ' + family)
        rows = []
        for text in native[family]:
            kind, *tokens = text.split(':')
            fields = {}
            for token in tokens:
                key, value = token.split('=', 1)
                fields.setdefault(key, []).append(value)
            rows.append({'descriptor': text, 'type': kind, 'fields': fields})
        result[family] = rows
    return result


def validate_ctr(native, expected, candidate):
    observed = ctr_evidence(native)
    simple = native['simple_ctr']
    if candidate:
        extra = [r for r in observed['simple_ctr'] if r['type'] == 'FloatTargetMeanValue']
        need(len(extra) == 1 and len(simple) == len(expected['simple_ctr']) + 1, 'Expected one additional Float CTR')
        need(sorted(r['descriptor'] for r in observed['simple_ctr'] if r['type'] != 'FloatTargetMeanValue')
             == sorted(expected['simple_ctr']), 'Original simple CTRs changed')
        fields = extra[0]['fields']
        for key, value in {'CtrBorderCount': ['15'], 'CtrBorderType': ['Uniform'], 'Prior': ['0/1']}.items():
            need(fields.get(key) == value, 'Float CTR option mismatch: ' + key)
        # A default TargetBorderCount in the Float description is retained as
        # evidence, not interpreted as discretization of its float sample.
    else:
        need(sorted(simple) == sorted(expected['simple_ctr']), 'Control simple CTR differs from saved R6 evidence')
    need(sorted(native['combinations_ctr']) == sorted(expected['combinations_ctr']), 'Combination CTR changed')
    for key in ('ctr_target_border_count', 'counter_calc_method', 'one_hot_max_size', 'max_ctr_complexity'):
        need(native.get(key) == expected[key], 'Unintended categorical setting change: ' + key)
    return observed


def model_json_summary(dump):
    """Map actual split_index to CTR type using official v1.2.10 bin order.

    Artificial input has only float/categorical features. Model JSON stores
    float borders, one-hot values, then CTR borders in the bin-feature order.
    The split's other fields are cross-checked; a name in params is not usage.
    """
    info = dump['features_info']
    need(not any(info.get(k) for k in ('text_features', 'embedding_features', 'estimated_features')),
         'Unexpected feature kind in artificial JSON dump')
    bins = []
    for f in info.get('float_features', []) or []:
        bins.extend({'split_type': 'FloatFeature', 'float_feature_index': f['feature_index'], 'border': b}
                    for b in f.get('borders', []))
    for f in sorted(info.get('categorical_features', []) or [], key=lambda x: x['feature_index']):
        bins.extend({'split_type': 'OneHotFeature', 'cat_feature_index': f['feature_index'], 'value': v}
                    for v in f.get('values', []) or [])
    ctrs = info.get('ctrs', []) or []
    for j, ctr in enumerate(ctrs):
        bins.extend({'split_type': 'OnlineCtr', 'border': b, 'ctr_target_border_idx': ctr['target_border_idx'],
                     'ctr_type': ctr['ctr_type'], 'ctr_feature_index': j}
                    for b in ctr['borders'])
    splits = []
    trees = dump['oblivious_trees']
    need(len(trees) == 2, 'Artificial JSON must have exactly two trees')
    for t, tree in enumerate(trees):
        for d, split in enumerate(tree.get('splits') or []):
            idx = split['split_index']
            need(type(idx) is int and 0 <= idx < len(bins), 'Unknown JSON bin-feature index')
            mapped = bins[idx]
            need(all(split.get(k) == v for k, v in mapped.items() if k not in ('ctr_type', 'ctr_feature_index')),
                 'JSON split does not match reconstructed bin ordering')
            splits.append(dict(mapped, tree=t, depth=d, split_index=idx))
    ctr_data_types = Counter(json.loads(k)['type'] for k in (dump.get('ctr_data') or {}))
    selected = [s for s in splits if s.get('ctr_type') == 'FloatTargetMeanValue']
    return {'ctr_features': ctrs, 'tree_splits': splits, 'ctr_data_type_counts': dict(ctr_data_types),
            'float_ctr_feature_count': sum(c['ctr_type'] == 'FloatTargetMeanValue' for c in ctrs),
            'float_ctr_selected_split_count': len(selected), 'float_ctr_selected_in_trees': bool(selected),
            'split_index_mapping_verified': True}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    reserved = ['feasibility.json', 'feasibility.sha256', 'events.jsonl']
    reserved += [name + suffix for name in NAMES for suffix in ('.cbm', '.model.json')]
    need(not any((OUT / n).exists() for n in reserved), 'Preserve existing artifacts; no repeat run')
    write_json(OUT / 'run_claim.json', {'started_at_utc': datetime.now(timezone.utc).isoformat(),
               'planned_fits': 2, 'source_sha256': sha(__file__), 'retry_allowed': False})
    with (OUT / 'events.jsonl').open('x', encoding='utf-8'):
        pass
    evidence = {'status': 'not_completed', 'synthetic_only': True, 'fits_started': 0, 'fits_completed': 0,
                'models_saved': 0, 'models_reloaded': 0, 'fully_verified_models': 0, 'models': {},
                'property_X_y_models_or_predictions_loaded': False, 'MAE_computed': False,
                'query_targets_generated': False, 'search_performed': False, 'alternate_settings_attempted': False,
                'data_seed': DATA_SEED, 'fit_rows': FIT_ROWS, 'query_rows': QUERY_ROWS,
                'feature_count': 95, 'categorical_count': 23, 'planned_fits': 2,
                'feasibility_overrides': {'iterations': 2, 'verbose': False},
                'candidate_passed_parameter_changes': ['simple_ctr'], 'extra_simple_ctr': EXTRA_CTR,
                'official_implementation_references': OFFICIAL_SOURCES,
                'runtime': {'catboost': catboost.__version__, 'numpy': np.__version__, 'pandas': pd.__version__}}
    phase = 'verify_fixed_inputs'
    try:
        need(catboost.__version__ == '1.2.10', 'This fixed probe targets local CatBoost 1.2.10')
        need(sha(ROOT / BASELINE) == BASELINE_SHA and sha(ROOT / REPORT) == REPORT_SHA, 'Research identity mismatch')
        receipt = (ROOT / BASELINE).with_suffix('.sha256')
        need(receipt.read_text(encoding='ascii').strip().split()[0] == BASELINE_SHA, 'Baseline receipt mismatch')
        original = json.loads((ROOT / BASELINE).read_text(encoding='utf-8'))
        need(sha(ROOT / CONFIG) == original['configuration']['sha256'], 'Original parameter config changed')
        evidence['source_sha256'] = {p: sha(ROOT / p) for p in SOURCES}
        for p, digest in original['source_sha256'].items():
            need(sha(ROOT / p) == digest, 'Baseline parameter source changed: ' + p)
        evidence['input_sha256'] = {BASELINE: BASELINE_SHA, receipt.relative_to(ROOT).as_posix(): sha(receipt),
                                    CONFIG: sha(ROOT / CONFIG), REPORT: REPORT_SHA,
                                    **original['source_sha256']}
        baseline = base.parameters()
        need(baseline == original['configuration']['effective_R6_passed_parameters'], 'R6 passed parameters changed')
        need(not any(k in baseline for k in ('simple_ctr', 'combinations_ctr', 'per_feature_ctr', 'ctr_target_border_count')),
             'Unexpected original CTR override')
        expected = original['all_native_categorical_option_fields']
        candidate = dict(baseline, simple_ctr=list(expected['simple_ctr']) + [EXTRA_CTR])
        evidence.update(baseline_parameters=baseline, candidate_parameters=candidate,
                        reference_native_categorical_settings=expected)
        view, y, cats = artificial_frames()
        frame_sha, target_sha = base.frame_sha(view), vector_sha(y)
        evidence.update(synthetic_feature_sha256=frame_sha, synthetic_fit_target_sha256=target_sha,
                        synthetic_fit_feature_sha256=base.frame_sha(view.iloc[:FIT_ROWS]),
                        synthetic_query_feature_sha256=base.frame_sha(view.iloc[FIT_ROWS:]),
                        artificial_target_formula='1 + (fit_row % 16)/4 + 0.01*(fit_row // 16)',
                        target_units='artificial scaled regression units; not observed property prices')
        for name, params in zip(NAMES, (baseline, candidate)):
            passed = dict(params, iterations=2, verbose=False)
            rec = {'status': 'constructed', 'passed_parameters': passed}
            evidence['models'][name] = rec
            phase = 'construct:' + name
            model = CatBoostRegressor(**passed)
            train = Pool(view.iloc[:FIT_ROWS], y, cat_features=cats, thread_count=4)
            query = Pool(view.iloc[FIT_ROWS:], cat_features=cats, thread_count=4)
            phase = 'fit:' + name
            evidence['fits_started'] += 1
            event({'event': 'fit_started', 'model': name, 'fit_number': evidence['fits_started'], 'trees': 2})
            started = time.monotonic()
            model.fit(train)
            evidence['fits_completed'] += 1
            rec.update(status='fitted', fit_seconds=time.monotonic()-started, fitted_native_parameters=model.get_all_params())
            event({'event': 'fit_completed', 'model': name, 'completed_count': evidence['fits_completed']})
            phase = 'save_and_reload:' + name
            path = OUT / (name + '.cbm')
            model.save_model(str(path))
            evidence['models_saved'] += 1
            rec.update(model_path=path.relative_to(ROOT).as_posix(), model_sha256=sha(path), status='saved')
            loaded = CatBoostRegressor()
            loaded.load_model(str(path))
            evidence['models_reloaded'] += 1
            rec.update(saved_native_parameters=loaded.get_all_params(), status='reloaded')
            dump_path = OUT / (name + '.model.json')
            loaded.save_model(str(dump_path), format='json')
            rec.update(model_json_path=dump_path.relative_to(ROOT).as_posix(), model_json_sha256=sha(dump_path))
            phase = 'verify_native_and_reload:' + name
            for label in ('fitted', 'saved'):
                native = rec[label + '_native_parameters']
                rec[label + '_ctr_evidence'] = ctr_evidence(native)
                validate_ctr(native, expected, name == NAMES[1])
                need(native['loss_function'] == native['eval_metric'] == 'MAE' and native['depth'] == 8
                     and native['boosting_type'] == 'Plain' and native['grow_policy'] == 'SymmetricTree'
                     and native['random_seed'] == 42 and native['iterations'] == 2 and native['task_type'] == 'GPU'
                     and np.isclose(native['learning_rate'], .065, rtol=1e-7, atol=0), 'Native base recipe differs')
            need(rec['fitted_ctr_evidence'] == rec['saved_ctr_evidence'], 'CTR config changed on serialization')
            need(loaded.tree_count_ == 2 and list(loaded.feature_names_) == list(view.columns)
                 and loaded.get_cat_feature_indices() == sorted(view.columns.get_loc(c) for c in cats), 'Native schema differs')
            a = np.asarray(model.predict(query, thread_count=4), dtype='<f8')
            b = np.asarray(loaded.predict(query, thread_count=4), dtype='<f8')
            need(a.shape == b.shape == (QUERY_ROWS,) and np.isfinite(a).all() and np.isfinite(b).all()
                 and a.tobytes() == b.tobytes(), 'Full artificial raw-query reload mismatch')
            rec.update(raw_reload_bitwise_equal=True, raw_prediction_sha256=vector_sha(a),
                       prediction_rows=QUERY_ROWS, configuration_verified=True)
            phase = 'map_actual_splits:' + name
            rec['model_json_summary'] = model_json_summary(json.loads(dump_path.read_text(encoding='utf-8')))
            rec['status'] = 'verified'
            evidence['fully_verified_models'] += 1
            event({'event': 'model_verified', 'model': name,
                   'float_ctr_selected_in_trees': rec['model_json_summary']['float_ctr_selected_in_trees']})
        control, changed = (evidence['models'][n]['saved_native_parameters'] for n in NAMES)
        differences = {k: {'control': control.get(k), 'candidate': changed.get(k)}
                       for k in sorted(set(control) | set(changed)) if control.get(k) != changed.get(k)}
        evidence['native_control_candidate_differences'] = differences
        need(set(differences) == {'simple_ctr'}, 'Unexpected native change beyond simple CTRs')
        need(base.frame_sha(view) == frame_sha and vector_sha(y) == target_sha, 'Artificial input mutated')
        for p, digest in {**evidence['input_sha256'], **evidence['source_sha256']}.items():
            need(sha(ROOT / p) == digest, 'Input/source changed during probe: ' + p)
        used = evidence['models'][NAMES[1]]['model_json_summary']['float_ctr_selected_in_trees']
        evidence.update(status='synthetic_GPU_configuration_verified', configuration_effect_verified=True,
                        float_ctr_selected_in_trees=used,
                        conclusion='registered_and_selected' if used else 'registered_but_not_selected_in_two_trees',
                        source_and_inputs_unchanged=True)
    except Exception as exc:
        evidence.update(status='synthetic_GPU_probe_failed', failure_phase=phase,
                        error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()})
    evidence.update(completed_at_utc=datetime.now(timezone.utc).isoformat(), no_retry_or_search_performed=True)
    evidence['events_sha256'] = sha(OUT / 'events.jsonl')
    write_json(OUT / 'feasibility.json', evidence)
    print(json.dumps({'status': evidence['status'], 'evidence_path': 'outputs/float_ctr_research/feasibility.json',
                      'evidence_sha256': sha(OUT / 'feasibility.json'), 'fits_started': evidence['fits_started'],
                      'fits_completed': evidence['fits_completed'], 'fully_verified_models': evidence['fully_verified_models'],
                      'float_ctr_selected_in_trees': evidence.get('float_ctr_selected_in_trees')}), flush=True)
    return 0 if evidence['status'] == 'synthetic_GPU_configuration_verified' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
