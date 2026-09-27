"""Evaluate four prespecified round-three candidates, then freeze one for confirmation.

Development: python src/round3_evaluate.py --split development [--freeze]
Confirmation: python src/round3_evaluate.py --split confirmation
Confirmation training happens AFTER --freeze. No training or scoring runs on import.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import importlib.metadata
import io
import json
import math
import pickle
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from model_common import ROOT, load_data, temporal_indices
from prepare_data import sha256
from round3_backtest import SPLITS
from round2_neighborhood import predict_reference
from round3_reference import fit_regional_trend, predict_regional_trend

OUT = ROOT / 'outputs' / 'round3' / 'evaluation'
FREEZE = ROOT / 'outputs' / 'round3' / 'frozen_candidates.json'
FREEZE_RECEIPT = FREEZE.with_suffix('.sha256')
GROUPS = ('seen', 'new_nonmissing', 'missing')
CONTROL = 'v2_control'
FORMULAS = {
    'v2_control': {'base': '0.75 * old_catboost + 0.25 * lightgbm_unit', 'reference': 'v2_trend'},
    'recency_blend': {'base': '0.75 * (0.5 * old_catboost + 0.5 * recency_catboost) + 0.25 * lightgbm_unit', 'reference': 'v2_trend'},
    'regional_trend': {'base': '0.75 * old_catboost + 0.25 * lightgbm_unit', 'reference': 'regional_trend'},
    'recency_regional': {'base': '0.75 * (0.5 * old_catboost + 0.5 * recency_catboost) + 0.25 * lightgbm_unit', 'reference': 'regional_trend'},
}
THRESHOLDS = {
    'development_min_overall_improvement': 0.01,
    'development_min_weighted_improvement': 0.01,
    'development_max_quarter_deterioration': 0.05,
    'large_stratum_min_rows': 500,
    'max_large_stratum_deterioration': 0.05,
    'confirmation_max_overall_deterioration': 0.0,
    'confirmation_max_weighted_deterioration': 0.0,
    'confirmation_min_improved_quarters': 2,
}
QUARTERS = {'Q1': [0, 91], 'Q2': [92, 183], 'Q3': [184, 275], 'Q4': [276, None]}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def json_default(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(type(value).__name__)


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True,
                       allow_nan=False, default=json_default) + '\n').encode('utf-8')


def relative(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def array_sha(values):
    return hashlib.sha256(np.asarray(values, dtype='<i8').tobytes()).hexdigest()


def verify_hashes(hashes):
    for path, digest in hashes.items():
        p = (ROOT / path).resolve()
        require(p.is_relative_to(ROOT.resolve()) and p.is_file(), f'Missing or unsafe frozen file: {path}')
        require(sha256(p) == digest, f'Frozen artifact changed: {path}')


def runtime_versions():
    return {name: importlib.metadata.version(name) for name in
            ['numpy', 'pandas', 'scipy', 'catboost', 'lightgbm', 'threadpoolctl']}


def source_hashes():
    """Trace local imports from fit/predict entrypoints; unrelated audit files may evolve."""
    pending = [ROOT / 'src' / name for name in
               ['round3_evaluate.py', 'round3_backtest.py', 'round3_recency.py', 'round3_reference.py']]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        require(path.is_file(), f'Fit/predict source is missing: {path}')
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                candidate = ROOT / 'src' / (name.split('.')[0] + '.py')
                if candidate.is_file() and candidate.resolve() not in found:
                    pending.append(candidate)
    return {relative(path): sha256(path) for path in sorted(found)}


def data_hashes():
    files = [ROOT / 'data_cache' / f'{split}{suffix}'
             for split in ['train', 'valid', 'test'] for suffix in ['.pkl', '_features.pkl', '.json']]
    files += list((ROOT / 'data_cache').glob('*features*.json'))
    files += [ROOT / 'runs/v1/models/final_config.json', ROOT / 'reports/feature_schema.json',
              ROOT / 'outputs/round3/candidate_plan.json', ROOT / 'outputs/round3/validation_protocol.json']
    for extra in ['requirements.txt', 'requirements-round2.txt']:
        if (ROOT / extra).is_file():
            files.append(ROOT / extra)
    require(all(path.is_file() for path in files), 'A required input cache/configuration is missing')
    return {relative(path): sha256(path) for path in sorted(set(files))}


def split_contract(data):
    result = {}
    for split, (start, end) in SPLITS.items():
        fi, ci = temporal_indices(data, start, end=end)
        require(not np.intersect1d(fi, ci).size, f'Fit/check overlap: {split}')
        result[split] = {'start': start, 'end_inclusive': end,
                         'fit_rows': len(fi), 'check_rows': len(ci),
                         'fit_indices_content_sha256_int64_le': array_sha(fi),
                         'check_indices_content_sha256_int64_le': array_sha(ci)}
    return result


def norm(series):
    values = series.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
    return values.mask(values.isin(['', 'nan', 'None', '<NA>', 'NaT']), '__MISSING__')


def project_keys(frame):
    project = norm(frame['project_name'])
    keys = norm(frame['address_city']) + '|' + norm(frame['address_district']) + '|' + project
    return keys, project.eq('__MISSING__')


def group_labels(fit, query):
    fk, fm = project_keys(fit)
    qk, qm = project_keys(query)
    known = set(fk[~fm])
    return np.where(qm, 'missing', np.where(qk.isin(known), 'seen', 'new_nonmissing'))


def target_weights(data):
    columns = ['address_city', 'address_district', 'project_name']
    fit = pd.concat([data['X_train'][columns], data['X_valid'][columns]], ignore_index=True)
    labels = group_labels(fit, data['X_test'])
    return {group: float(np.mean(labels == group)) for group in GROUPS}


def artifact_hashes(folder):
    allowed = {'.npy', '.pkl', '.json', '.cbm', '.txt'}
    return {relative(p): sha256(p) for p in sorted(folder.rglob('*'))
            if p.is_file() and p.suffix in allowed and 'catboost_info' not in p.parts}


def parameters(metadata):
    result = metadata.get('parameters', metadata.get('params'))
    require(isinstance(result, dict), 'Training metadata has no parameter dictionary')
    if 'params' in metadata and 'parameters' in metadata:
        require(metadata['params'] == metadata['parameters'], 'Parameter aliases differ')
    return result


def validate_inputs(data, split, need_recency, expected_parameters=None):
    start, end = SPLITS[split]
    fi, ci = temporal_indices(data, start, end=end)
    folder = ROOT / 'outputs/round3/baseline' / split
    for filename, expected in [('fit_indices.npy', fi), ('check_indices.npy', ci)]:
        require(np.array_equal(np.load(folder / filename, allow_pickle=False), expected), f'Baseline index order differs: {filename}')
    hashes = artifact_hashes(folder)
    train_parameters = {}
    for name in ['catboost_raw', 'lightgbm_l1_unit63', 'v2_control']:
        metadata = read(folder / f'{name}.json')
        require(metadata['split'] == split and metadata['start'] == start and metadata['end_inclusive'] == end, f'Wrong baseline split: {name}')
        require(metadata['fit_rows'] == len(fi) and metadata['check_rows'] == len(ci), f'Wrong baseline rows: {name}')
        require(metadata['fit_uses_eval_set'] is False and metadata['iterations_selected_from_confirmation'] is False, f'Unexpected check-label tuning: {name}')
        require(metadata['script_sha256'] == sha256(ROOT / 'src/round3_backtest.py'), f'Baseline training source differs: {name}')
        require(metadata['feature_source_sha256'] == sha256(ROOT / 'src/house_features.py'), f'Baseline feature source differs: {name}')
        require(metadata['source_sha256'] == data['metadata']['source_sha256'], f'Baseline inputs differ: {name}')
        require(metadata['model_sha256'] == sha256(ROOT / metadata['model_path']), f'Baseline model differs: {name}')
        require(metadata['prediction_sha256'] == sha256(folder / f'{name}.npy'), f'Baseline predictions differ: {name}')
        for field, filename in [('fit_indices_sha256', 'fit_indices.npy'), ('check_indices_sha256', 'check_indices.npy')]:
            if field in metadata:
                require(metadata[field] == sha256(folder / filename), f'Baseline index SHA differs: {name}/{field}')
        train_parameters[name] = parameters(metadata)
    if need_recency:
        recency = ROOT / 'outputs/round3/recency' / split
        metadata = read(recency / 'metadata.json')
        require(metadata['split'] == split and metadata['start'] == start and metadata['end_inclusive'] == end, 'Wrong recency split')
        require(metadata['fit_rows'] == len(fi) and metadata['query_rows'] == len(ci), 'Wrong recency row counts')
        require(metadata['fit_uses_eval_set'] is False, 'Recency used a check eval_set')
        require(metadata['source_sha256'] == data['metadata']['source_sha256'], 'Recency input sources differ')
        require(metadata['script_sha256'] == sha256(ROOT / 'src/round3_recency.py'), 'Recency source differs')
        require(metadata['feature_source_sha256'] == sha256(ROOT / 'src/house_features.py'), 'Recency feature source differs')
        require(metadata['feature_schema_sha256'] == sha256(ROOT / 'reports/feature_schema.json'), 'Recency schema differs')
        verify_hashes(metadata['dependency_script_sha256'])
        for filename, expected, field in [('fit_indices.npy', fi, 'fit_indices_sha256'), ('query_indices.npy', ci, 'query_indices_sha256')]:
            require(np.array_equal(np.load(recency / filename, allow_pickle=False), expected), f'Recency index order differs: {filename}')
            require(metadata[field] == sha256(recency / filename), f'Recency index SHA differs: {filename}')
        require(metadata['model_sha256'] == sha256(recency / 'model.cbm'), 'Recency model differs')
        require(metadata['prediction_sha256'] == sha256(recency / 'predictions.npy'), 'Recency predictions differ')
        train_parameters['recency_catboost'] = parameters(metadata)
        canonical = json.dumps(train_parameters['recency_catboost'], sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
        require(metadata['parameters_sha256'] == hashlib.sha256(canonical).hexdigest(), 'Recency parameter SHA differs')
        hashes.update(artifact_hashes(recency))
    if expected_parameters is not None:
        for name, value in train_parameters.items():
            require(value == expected_parameters[name], f'Frozen training parameters changed: {name}')
    return fi, ci, hashes, train_parameters


def load_vector(path, count):
    values = np.asarray(np.load(path, allow_pickle=False), dtype='float64')
    require(values.shape == (count,) and np.isfinite(values).all() and (values >= 0).all(), f'Invalid predictions: {path}')
    return values


def predict_candidates(data, split, names, fi, ci, frozen=None):
    """Only fitting-side y enters the regional fit; query y is not accessed here."""
    folder = ROOT / 'outputs/round3/baseline' / split
    xf = data['X_train'].iloc[fi].reset_index(drop=True)
    xq = data['X_train'].iloc[ci].reset_index(drop=True)
    with (folder / 'reference_v2.pkl').open('rb') as handle:
        reference = pickle.load(handle)
    require(reference['fit_report']['fitting_rows'] == len(fi), 'Reference fitting-row count differs')
    maximum_day = float(np.nanmax(xf['trade_day_index'].to_numpy(dtype=float)))
    require(reference['fit_report']['hedonic']['maximum_fit_day_index'] == maximum_day, 'Reference training date anchor differs')
    cat = load_vector(folder / 'catboost_raw.npy', len(ci))
    unit = load_vector(folder / 'lightgbm_l1_unit63.npy', len(ci))
    base = .75 * cat + .25 * unit
    recent_base = None
    if any(name.startswith('recency') for name in names):
        recent = load_vector(ROOT / 'outputs/round3/recency' / split / 'predictions.npy', len(ci))
        recent_base = .75 * (.5 * cat + .5 * recent) + .25 * unit
    regional = None
    if any('regional' in name for name in names):
        regional = fit_regional_trend(xf, np.asarray(data['y_train'])[fi], reference)
        require(regional['training_rows'] == len(fi), 'Regional fitting-row count differs')
        if frozen is not None:
            require(json.loads(encoded(regional['parameters'])) == frozen['regional_parameters'], 'Regional parameters changed after freeze')
    predictions = {}
    for name in names:
        baseline = recent_base if name.startswith('recency') else base
        if 'regional' in name:
            values = predict_regional_trend(reference, regional, xq, baseline)
        else:
            values = predict_reference(reference, xq, baseline, variant='trend')
        values = np.asarray(values, dtype='float64')
        require(values.shape == (len(ci),) and np.isfinite(values).all() and (values >= 0).all(), f'Invalid candidate: {name}')
        predictions[name] = values
    require(np.allclose(predictions[CONTROL], load_vector(folder / 'v2_control.npy', len(ci)), rtol=1e-12, atol=1e-6), 'Recomputed v2 control differs from saved control')
    return predictions, regional


def score(y, prediction, fit, query, start, weights):
    errors = np.abs(np.asarray(y, dtype='float64') - prediction)
    labels = group_labels(fit, query)
    day0 = (pd.Timestamp(start) - pd.Timestamp('2000-01-01')).days
    horizon = query['trade_day_index'].to_numpy(dtype='float64') - day0
    require(np.isfinite(horizon).all() and (horizon >= 0).all(), 'Invalid forecast horizon')
    month = pd.to_datetime(query['trade_day_index'], unit='D', origin='2000-01-01').dt.strftime('%Y-%m').to_numpy()
    city = norm(query['address_city']).to_numpy()
    def part(mask):
        n = int(np.sum(mask))
        return {'rows': n, 'mae': float(errors[mask].mean()) if n else None}
    strata = {group: part(labels == group) for group in GROUPS}
    weighted = 0.0
    for group, weight in weights.items():
        if weight > 0:
            require(strata[group]['rows'] > 0, f'Positive-weight stratum has no validation rows: {group}')
            weighted += weight * strata[group]['mae']
    quarters = {}
    for name, (low, high) in QUARTERS.items():
        mask = horizon >= low
        if high is not None:
            mask &= horizon <= high
        quarters[name] = {**part(mask), 'horizon_days': [low, high]}
    return {'rows': len(errors), 'overall_mae': float(errors.mean()), 'weighted_mae': float(weighted),
            'strata': strata, 'forecast_quarters': quarters,
            'months': {str(key): part(month == key) for key in sorted(set(month))},
            'cities': {str(key): part(city == key) for key in sorted(set(city))}}


def development_selection(metrics):
    base = metrics[CONTROL]
    checks = {}
    qualified = []
    for name, value in metrics.items():
        if name == CONTROL:
            continue
        rules = {
            'overall_improves_at_least_1pct': value['overall_mae'] <= .99 * base['overall_mae'],
            'weighted_improves_at_least_1pct': value['weighted_mae'] <= .99 * base['weighted_mae'],
            'all_quarters_deteriorate_at_most_5pct': all(value['forecast_quarters'][q]['mae'] <= 1.05 * b['mae'] for q,b in base['forecast_quarters'].items() if b['rows']),
            'large_strata_deteriorate_at_most_5pct': all(value['strata'][g]['mae'] <= 1.05 * b['mae'] for g,b in base['strata'].items() if b['rows'] >= 500),
        }
        eligible = all(rules.values())
        checks[name] = {'eligible': eligible, 'checks': rules}
        if eligible:
            qualified.append(name)
    selected = min(qualified, key=lambda name: (metrics[name]['weighted_mae'], name)) if qualified else CONTROL
    return {'selected_candidate': selected, 'candidate_checks': checks,
            'qualified_candidates': qualified, 'thresholds': THRESHOLDS,
            'selection_rule': 'Among eligible candidates choose lowest weighted MAE; deterministic name tie-break; otherwise retain control.'}


def confirmation_decision(metrics, selected):
    base = metrics[CONTROL]
    value = metrics[selected]
    improved_quarters = [q for q,b in base['forecast_quarters'].items() if b['rows'] and value['forecast_quarters'][q]['mae'] < b['mae']]
    rules = {
        'overall_not_worse': value['overall_mae'] <= base['overall_mae'],
        'weighted_not_worse': value['weighted_mae'] <= base['weighted_mae'],
        'at_least_two_quarters_improve': len(improved_quarters) >= 2,
        'large_strata_deteriorate_at_most_5pct': all(value['strata'][g]['mae'] <= 1.05 * b['mae'] for g,b in base['strata'].items() if b['rows'] >= 500),
    }
    accepted = selected != CONTROL and all(rules.values())
    return {'selected_candidate': selected, 'final_accept': accepted, 'checks': rules,
            'improved_quarters': improved_quarters,
            'deployment_decision': selected if accepted else 'retain_v2',
            'candidate_reselection_performed': False}


def verify_saved_report(report):
    verify_hashes(report['source_sha256'])
    verify_hashes(report['data_sha256'])
    verify_hashes(report['input_artifact_sha256'])
    verify_hashes(report['output_artifact_sha256'])
    require(report['candidate_formulas'] == FORMULAS and report['thresholds'] == THRESHOLDS, 'Evaluation contract changed')
    require(report['runtime_versions'] == runtime_versions(), 'Runtime versions changed')


def evaluate(split, frozen=None):
    names = list(FORMULAS) if split == 'development' else list(dict.fromkeys([CONTROL, frozen['selected_candidate']]))
    target = OUT / split
    require(not target.exists(), f'Evaluation directory already exists: {target}; old results are never overwritten')
    sources, inputs = source_hashes(), data_hashes()
    data = load_data()
    contract = split_contract(data)
    weights = target_weights(data)
    if frozen is not None:
        require(contract == frozen['splits'] and weights == frozen['target_group_weights'], 'Frozen split or test group weights changed')
        require(sources == frozen['source_sha256'] and inputs == frozen['data_sha256'], 'Frozen source or data contract changed')
    fi, ci, model_hashes, train_parameters = validate_inputs(
        data, split, any(name.startswith('recency') for name in names),
        None if frozen is None else frozen['training_parameters'])
    # Provenance is validated before any confirmation prediction or query-label scoring.
    if frozen is not None:
        freeze_epoch = datetime.fromisoformat(frozen['freeze_timestamp_utc']).timestamp()
        require(frozen['confirmation_artifacts_absent_at_freeze'] is True, 'No pre-training freeze evidence')
        require(list(data['X_train'].columns) == frozen['feature_columns'], 'Frozen feature order changed')
        for path in model_hashes:
            require((ROOT / path).stat().st_mtime >= freeze_epoch, f'Confirmation artifact predates freeze: {path}')
        if any(name.startswith('recency') for name in names):
            recency_meta = read(ROOT / 'outputs/round3/recency' / split / 'metadata.json')
            for key in ['frozen_candidates_sha256', 'freeze_sha256']:
                if key in recency_meta and recency_meta[key] is not None:
                    require(recency_meta[key] == sha256(FREEZE), 'Recency was trained under another freeze')
    predictions, regional = predict_candidates(data, split, names, fi, ci, frozen)
    fit, query = data['X_train'].iloc[fi], data['X_train'].iloc[ci]
    query_y = np.asarray(data['y_train'])[ci]
    metrics = {name: score(query_y, p, fit, query, SPLITS[split][0], weights) for name,p in predictions.items()}
    selection = development_selection(metrics) if split == 'development' else confirmation_decision(metrics, frozen['selected_candidate'])
    verify_hashes(sources); verify_hashes(inputs); verify_hashes(model_hashes)
    OUT.mkdir(parents=True, exist_ok=True)
    stage = OUT / f'.{split}-staging-{uuid.uuid4().hex}'
    stage.mkdir()
    try:
        for name, prediction in predictions.items():
            np.save(stage / f'{name}.npy', prediction, allow_pickle=False)
            metadata = {'candidate': name, 'formula': FORMULAS[name], 'split': split,
                        'split_contract': contract[split], 'metrics': metrics[name],
                        'target_group_weights': weights, 'prediction_units': 'TWD', 'rounded': False,
                        'source_sha256': sources, 'input_artifact_sha256': model_hashes,
                        'query_targets_used_for_prediction': False}
            metadata['prediction_sha256'] = sha256(stage / f'{name}.npy')
            (stage / f'{name}.json').write_bytes(encoded(metadata))
        if regional is not None:
            (stage / 'regional_model.pkl').write_bytes(pickle.dumps(regional, protocol=pickle.HIGHEST_PROTOCOL))
            (stage / 'regional_model.json').write_bytes(encoded(regional))
        with (stage / 'comparison.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            fields = ['candidate','overall_mae','weighted_mae'] + [f'{g}_mae' for g in GROUPS] + [f'{q}_mae' for q in QUARTERS]
            writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
            for name,m in metrics.items():
                writer.writerow({'candidate':name,'overall_mae':m['overall_mae'],'weighted_mae':m['weighted_mae'],
                                 **{f'{g}_mae':m['strata'][g]['mae'] for g in GROUPS},
                                 **{f'{q}_mae':m['forecast_quarters'][q]['mae'] for q in QUARTERS}})
        output_hashes = {relative(target / path.name):sha256(path) for path in sorted(stage.iterdir()) if path.is_file()}
        report = {'format_version':1,'split':split,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                  'candidate_formulas':FORMULAS,'evaluated_candidates':names,'thresholds':THRESHOLDS,
                  'splits':contract,'feature_columns':list(data['X_train'].columns),
                  'project_key':'address_city | address_district | project_name; missing is a separate stratum',
                  'target_group_weights':weights,'metric':'MAE','unit':'TWD',
                  'metrics':metrics,'selection':selection,'source_sha256':sources,'data_sha256':inputs,
                  'input_artifact_sha256':model_hashes,'output_artifact_sha256':output_hashes,
                  'training_parameters':train_parameters,'regional_parameters':None if regional is None else regional['parameters'],
                  'runtime_versions':runtime_versions(),'query_targets_used_for_prediction':False,
                  'public_score_used_as_target':False,'historically_untouched_confirmation':False,
                  'indices_verified_by_content_hash':True}
        if frozen is not None:
            report['frozen_candidates_sha256'] = sha256(FREEZE)
            report['confirmation_artifacts_created_after_freeze'] = True
            report['final_accept'] = selection['final_accept']
            # The freeze is immutable; new confirmation input hashes live only in this result.
        (stage / 'results.json').write_bytes(encoded(report))
        require(not target.exists(), 'Evaluation appeared during processing; refusing overwrite')
        require(stage.resolve().parent == OUT.resolve() and target.resolve().parent == OUT.resolve(), 'Unsafe evaluation move')
        stage.rename(target)
    finally:
        if stage.exists():
            resolved = stage.resolve()
            require(resolved.parent == OUT.resolve() and resolved.name.startswith(f'.{split}-staging-') and resolved.is_relative_to(ROOT.resolve()), 'Unsafe staging cleanup')
            shutil.rmtree(resolved)
    return report


def freeze_development(report):
    require(report['split'] == 'development', 'Only development can choose and freeze candidates')
    verify_saved_report(report)
    data = load_data()
    require(split_contract(data) == report['splits'] and target_weights(data) == report['target_group_weights'], 'Split contract changed before freeze')
    for split in ['confirmation']:
        # Prevent claiming a freeze before already materialized confirmation models/scores.
        require(not (ROOT / 'outputs/round3/baseline' / split).exists(), 'Confirmation baseline exists before freeze')
        require(not (ROOT / 'outputs/round3/recency' / split).exists(), 'Confirmation recency exists before freeze')
    path = OUT / 'development/results.json'
    content = {'format_version':1,'freeze_timestamp_utc':datetime.now(timezone.utc).isoformat(),
               'selected_candidate':report['selection']['selected_candidate'],
               'confirmation_artifacts_absent_at_freeze':True,
               'candidate_formulas':FORMULAS,'thresholds':THRESHOLDS,'forecast_quarters':QUARTERS,
               'source_sha256':report['source_sha256'],'data_sha256':report['data_sha256'],
               'splits':report['splits'],'feature_columns':report['feature_columns'],
               'target_group_weights':report['target_group_weights'],
               'training_parameters':report['training_parameters'],'regional_parameters':report['regional_parameters'],
               'runtime_versions':report['runtime_versions'],
               'development_model_and_prediction_sha256':report['input_artifact_sha256'],
               'development_evaluation_artifact_sha256':{**report['output_artifact_sha256'],relative(path):sha256(path)},
               'selection':report['selection'],
               'confirmation_policy':'Train only after this freeze; validate source/data/params and precommitted index content hashes, then evaluate control plus the single selected candidate. Confirmation model/prediction hashes are recorded in confirmation results without changing this file.',
               'public_score_used_as_target':False,'historically_untouched_confirmation':False}
    FREEZE.parent.mkdir(parents=True, exist_ok=True)
    require(not FREEZE_RECEIPT.exists(), 'Freeze hash receipt already exists')
    with FREEZE.open('xb') as handle:
        handle.write(encoded(content))
    with FREEZE_RECEIPT.open('x', encoding='ascii') as handle:
        handle.write(sha256(FREEZE) + '\n')
    return content


def verify_freeze():
    require(FREEZE.is_file(), 'Freeze development before confirmation')
    require(FREEZE_RECEIPT.is_file() and FREEZE_RECEIPT.read_text(encoding='ascii').strip() == sha256(FREEZE), 'Frozen-candidate SHA receipt is missing or differs')
    frozen = read(FREEZE)
    require(frozen['selected_candidate'] in FORMULAS, 'Unknown frozen candidate')
    require(frozen['candidate_formulas'] == FORMULAS and frozen['thresholds'] == THRESHOLDS and frozen['forecast_quarters'] == QUARTERS, 'Frozen candidate/rule contract differs')
    require(frozen['runtime_versions'] == runtime_versions(), 'Frozen runtime changed')
    for key in ['source_sha256','data_sha256','development_model_and_prediction_sha256','development_evaluation_artifact_sha256']:
        verify_hashes(frozen[key])
    development = read(OUT / 'development/results.json')
    require(frozen['selection'] == development['selection'], 'Frozen development selection differs')
    require(frozen['selected_candidate'] == development['selection']['selected_candidate'], 'Frozen candidate is not the automatically selected candidate')
    return frozen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--split', choices=list(SPLITS), required=True)
    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()
    require(not args.freeze or args.split == 'development', '--freeze is only valid for development')
    if args.split == 'confirmation':
        frozen = verify_freeze()
        report = evaluate('confirmation', frozen)
    else:
        require(not FREEZE.exists(), 'Candidates are already frozen; do not reopen development tuning')
        saved = OUT / 'development/results.json'
        report = read(saved) if saved.is_file() else evaluate('development')
        verify_saved_report(report)
        if args.freeze:
            frozen = freeze_development(report)
            print(json.dumps({'event':'candidates_frozen','selected_candidate':frozen['selected_candidate'],'path':str(FREEZE)},ensure_ascii=True),flush=True)
    print(json.dumps({'split':args.split,'evaluated_candidates':report['evaluated_candidates'],
                      'selection':report['selection'],'results':str(OUT/args.split/'results.json')},ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()