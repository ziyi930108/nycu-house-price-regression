"""R23 fixed artificial GPU CTR-target-border compatibility test.

No property X/y, prediction, or model is read. The R6 v1 JSON is read for passed
parameters only. Exactly two artificial fits (two trees each) are attempted:
original parameters and global ctr_target_border_count=8. There is no search,
price score, alternate value, retry, or fallback. Execute only with --run.
The caller captures stdout/stderr to ctr_feasibility.log exclusively.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time
import traceback
import numpy as np
import pandas as pd
import catboost
from catboost import CatBoostRegressor, Pool
import round6_geo as base

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round23_research'
SEED = 20260925
CANDIDATE_TARGET_BORDER_COUNT = 8
FIT_ROWS, QUERY_ROWS = 512, 32
MODEL_NAMES = ('mae_control', 'ctr8_candidate')
SOURCE_NAMES = ('src/round23_ctr_feasibility.py', 'src/round6_geo.py', 'src/prepare_data.py')
CONFIG_PATH = 'runs/v1/models/final_config.json'


def need(value, message):
    if not value:
        raise ValueError(message)


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            result.update(block)
    return result.hexdigest()


def vector_sha(value):
    return hashlib.sha256(np.asarray(value, dtype='<f8').tobytes(order='C')).hexdigest()


def save_evidence(path, value):
    encoded = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                          allow_nan=False) + '\n').encode('utf-8')
    with path.open('xb') as handle:
        handle.write(encoded)
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(hashlib.sha256(encoded).hexdigest() + '\n')


def parse_native_ctr(native):
    """Preserve every descriptor; target-free CTRs have a null count, not zero."""
    result = {'global_ctr_target_border_count': native.get('ctr_target_border_count'),
              'simple_ctr': [], 'combinations_ctr': [], 'target_count_values': {}}
    global_count = result['global_ctr_target_border_count']
    need(type(global_count) is int and global_count > 0, 'Missing/invalid native global target-border count')
    for family in ('simple_ctr', 'combinations_ctr'):
        descriptors = native[family]
        need(isinstance(descriptors, list) and descriptors, 'Missing native CTR descriptors: ' + family)
        counts = []
        for descriptor in descriptors:
            need(isinstance(descriptor, str), 'Unexpected native CTR descriptor type')
            tokens = descriptor.split(':')
            values = [token.split('=', 1)[1] for token in tokens if token.startswith('TargetBorderCount=')]
            need(len(values) <= 1, 'Duplicate TargetBorderCount in one native descriptor')
            count = None
            if values:
                need(values[0].isdigit() and int(values[0]) > 0, 'Invalid native TargetBorderCount')
                count = int(values[0]); counts.append(count)
            result[family].append({'descriptor': descriptor, 'ctr_type': tokens[0],
                                   'target_border_count': count})
        result['target_count_values'][family] = sorted(set(counts))
    return result


def require_global_ctr_applied(observed, expected=None):
    count = observed['global_ctr_target_border_count']
    if expected is not None:
        need(count == expected, 'Requested global CTR target-border count was not retained natively')
    for family in ('simple_ctr', 'combinations_ctr'):
        need(observed['target_count_values'][family] == [count],
             'Target-based native ' + family + ' does not reflect the global target-border count')


def artificial_frames():
    rng = np.random.default_rng(SEED)
    n = FIT_ROWS + QUERY_ROWS
    cats = ['category_' + str(i) for i in range(21)]
    X = pd.DataFrame({c: pd.Series(['g' + str((row+i) % (11+i % 3)) for row in range(n)],
                                  dtype=object) for i, c in enumerate(cats)})
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
    need(view.shape == (544, 95) and len(categories) == 23, 'Wrong fixed artificial schema')
    y = (2. + .25 * np.tanh(X.numeric_0.to_numpy(dtype='float64'))
         + .5 * np.maximum(X.numeric_1.to_numpy(dtype='float64'), 0)
         + (np.arange(n) % 19 == 0) * 3.)
    need(np.isfinite(y).all() and (y >= 0).all(), 'Invalid artificial target')
    return view, y, categories


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / 'ctr_feasibility.json'
    owned = [path, path.with_suffix('.sha256')] + [OUT / (name + '.cbm') for name in MODEL_NAMES]
    need(not any(p.exists() for p in owned), 'Existing compatibility evidence/model; preserve and do not rerun')
    source_hashes = {name: sha(ROOT / name) for name in SOURCE_NAMES}
    config_hash = sha(ROOT / CONFIG_PATH)
    evidence = {
        'round': 23, 'status': 'not_completed', 'recorded_at_utc': datetime.now(timezone.utc).isoformat(),
        'synthetic_only': True, 'property_X_y_models_or_predictions_loaded': False,
        'property_MAE_computed': False, 'baseline_configuration_read_for_parameters_only': True,
        'synthetic_seed': SEED, 'planned_synthetic_model_count': 2, 'synthetic_model_count': 0,
        'feasibility_overrides': {'iterations': 2, 'verbose': False},
        'declared_candidate_ctr_target_border_count': CANDIDATE_TARGET_BORDER_COUNT,
        'candidate_passed_parameter_changes': ['ctr_target_border_count'],
        'border_values_searched': False, 'alternate_values_attempted': False,
        'input_sha256': {CONFIG_PATH: config_hash}, 'source_sha256': source_hashes, 'models': {},
        'runtime': {'catboost': catboost.__version__, 'numpy': np.__version__, 'pandas': pd.__version__},
        'documentation_question': {
            'url': 'https://catboost.ai/docs/en/references/training-parameters/ctr',
            'scope': 'Verify this local GPU/global-parameter behavior; do not infer embedded TargetBorderCount argument support.'},
        'limitations': [
            'Artificial two-tree compatibility does not establish price accuracy, formal 2999-tree success, cost, or generalization.',
            'The only candidate parameter value is global ctr_target_border_count=8; this is not a border search.',
            'Saved native CTR specifications show configuration; two trees do not prove every category used target-based CTR splits.',
            'Target-free CTR descriptors have no TargetBorderCount and are recorded as null, never treated as zero.',
            'No prediction error or MAE is calculated, even for artificial query labels.',
            'The untouched original R6 parameters are read from the v1 configuration; no original model is loaded.'],
    }
    phase = 'read_parameters'
    try:
        baseline = base.parameters()
        need(baseline['loss_function'] == baseline['eval_metric'] == 'MAE'
             and baseline['iterations'] == 2999 and baseline['learning_rate'] == .065
             and baseline['task_type'] == 'GPU' and baseline['depth'] == 8
             and baseline['boosting_type'] == 'Plain' and baseline['thread_count'] == 4,
             'Wrong fixed R6 GPU/MAE parameters')
        need(not any(k in baseline for k in ('simple_ctr', 'combinations_ctr', 'per_feature_ctr')),
             'Unexpected passed per-CTR overrides; do not silently replace them')
        candidate = dict(baseline, ctr_target_border_count=CANDIDATE_TARGET_BORDER_COUNT)
        changed = sorted(k for k in set(baseline) | set(candidate) if baseline.get(k) != candidate.get(k))
        need(changed == ['ctr_target_border_count'], 'Only the global target-border parameter may change')
        evidence.update(baseline_parameters=baseline, candidate_parameters=candidate,
            baseline_ctr_target_border_count_explicit=('ctr_target_border_count' in baseline))
        phase = 'make_artificial_frames'
        view, y, categories = artificial_frames()
        feature_sha, target_sha = base.frame_sha(view), vector_sha(y)
        evidence.update(synthetic_feature_sha256=feature_sha, synthetic_target_sha256=target_sha,
            synthetic_fit_feature_sha256=base.frame_sha(view.iloc[:FIT_ROWS]),
            synthetic_query_feature_sha256=base.frame_sha(view.iloc[FIT_ROWS:]),
            synthetic_fit_target_sha256=vector_sha(y[:FIT_ROWS]),
            fit_rows=FIT_ROWS, query_rows=QUERY_ROWS, feature_count=95, categorical_feature_count=23,
            query_labels_used_for_training=False)
        for name, params in [('mae_control', baseline), ('ctr8_candidate', candidate)]:
            record = {'status': 'started', 'fit_rows': FIT_ROWS, 'query_rows': QUERY_ROWS}
            evidence['models'][name] = record
            passed = dict(params, iterations=2, verbose=False)
            record['passed_parameters'] = passed
            phase = 'construct_and_fit:' + name
            print(json.dumps({'event': 'artificial_fit_started', 'model': name, 'trees': 2}), flush=True)
            model = CatBoostRegressor(**passed)
            started = time.monotonic()
            model.fit(Pool(view.iloc[:FIT_ROWS], y[:FIT_ROWS], cat_features=categories, thread_count=4))
            record['fitted_native_parameters'] = model.get_all_params()
            record['fitted_ctr_evidence'] = parse_native_ctr(record['fitted_native_parameters'])
            phase = 'save_and_reload:' + name
            model_path = OUT / (name + '.cbm')
            need(not model_path.exists(), 'Artificial model destination already exists')
            model.save_model(str(model_path))
            record.update(model_path=model_path.relative_to(ROOT).as_posix(), model_sha256=sha(model_path))
            loaded = CatBoostRegressor(); loaded.load_model(str(model_path))
            native = loaded.get_all_params()
            record['saved_native_parameters'] = native
            record['saved_ctr_evidence'] = parse_native_ctr(native)
            record['native_values_changed_during_save'] = {
                k: {'fitted': record['fitted_native_parameters'].get(k), 'saved': native.get(k)}
                for k in sorted(set(record['fitted_native_parameters']) | set(native))
                if record['fitted_native_parameters'].get(k) != native.get(k)}
            phase = 'verify_artificial_model:' + name
            query = Pool(view.iloc[FIT_ROWS:], cat_features=categories, thread_count=4)
            before = np.asarray(model.predict(query, thread_count=4), dtype='float64')
            after = np.asarray(loaded.predict(query, thread_count=4), dtype='float64')
            need(np.isfinite(before).all() and before.shape == (QUERY_ROWS,)
                 and np.array_equal(before, after), 'Artificial saved/reloaded predictions differ')
            need(loaded.tree_count_ == 2 and list(loaded.feature_names_) == list(view.columns)
                 and loaded.get_cat_feature_indices() == sorted(view.columns.get_loc(c) for c in categories),
                 'Artificial native feature/tree/category schema differs')
            need(native['loss_function'] == native['eval_metric'] == 'MAE' and native['boosting_type'] == 'Plain'
                 and native['depth'] == 8 and native['grow_policy'] == 'SymmetricTree'
                 and np.isclose(native['learning_rate'], .065, rtol=1e-7, atol=0), 'Artificial native base recipe differs')
            wanted = CANDIDATE_TARGET_BORDER_COUNT if name == 'ctr8_candidate' else None
            for field in ('fitted_ctr_evidence', 'saved_ctr_evidence'):
                require_global_ctr_applied(record[field], wanted)
            need(record['fitted_ctr_evidence'] == record['saved_ctr_evidence'], 'CTR specifications changed in serialization')
            record.update(status='passed', tree_count=int(loaded.tree_count_), feature_count=len(loaded.feature_names_),
                categorical_positions=loaded.get_cat_feature_indices(), saved_reload_predictions_exact=True,
                synthetic_query_prediction_sha256=vector_sha(before), elapsed_seconds=time.monotonic()-started)
            evidence['synthetic_model_count'] += 1
            print(json.dumps({'event': 'artificial_model_verified', 'model': name,
                'ctr_evidence': record['saved_ctr_evidence']}), flush=True)
        phase = 'verify_inputs_and_sources'
        need(base.frame_sha(view) == feature_sha and vector_sha(y) == target_sha, 'Artificial frame/target mutated')
        a, b = (evidence['models'][name]['saved_native_parameters'] for name in MODEL_NAMES)
        evidence['native_control_candidate_differences'] = {
            k: {'control': a.get(k), 'candidate': b.get(k)} for k in sorted(set(a) | set(b)) if a.get(k) != b.get(k)}
        need({name: sha(ROOT / name) for name in SOURCE_NAMES} == source_hashes
             and sha(ROOT / CONFIG_PATH) == config_hash, 'Source/config changed during artificial run')
        evidence.update(status='synthetic_GPU_compatibility_passed', source_and_config_unchanged=True)
    except Exception as exc:
        evidence.update(status='synthetic_GPU_compatibility_failed', failure_phase=phase,
            error={'type': type(exc).__name__, 'message': str(exc), 'traceback': traceback.format_exc()},
            no_alternative_value_or_retry_attempted=True)
    evidence['completed_at_utc'] = datetime.now(timezone.utc).isoformat()
    save_evidence(path, evidence)
    print(json.dumps({'status': evidence['status'], 'evidence_path': path.relative_to(ROOT).as_posix(),
        'evidence_sha256': sha(path), 'completed_synthetic_models': evidence['synthetic_model_count'],
        'failure_phase': evidence.get('failure_phase'), 'no_price_score_computed': True}), flush=True)
    return 0 if evidence['status'] == 'synthetic_GPU_compatibility_passed' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', action='store_true', required=True)
    parser.parse_args()
    raise SystemExit(run())
