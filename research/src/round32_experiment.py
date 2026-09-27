"""One predeclared XGBoost geography model; original R6 data and price gates.

Commands are explicit. Importing this file never reads housing data or trains.
Saved predictions are checked against the saved model once in fit; evaluation
then verifies those artifacts, original indices, inverse, and the v3 control.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time

import numpy as np
import round32_xgboost_geo as component
import round6_common as common
import round6_evaluate as evaluation
from round6_geo import frame_sha
from round6_prepare_geo import load_geographic_partition
from round2_neighborhood import predict_reference
from round3_evaluate import score, target_weights

ROOT = common.ROOT
OUT = ROOT / 'outputs/round32'
STAGES = tuple(common.SPLITS)
CONTROL = 'v3_control'
CANDIDATE = 'xgboost_geography_blend'
THRESHOLDS = dict(evaluation.THRESHOLDS)
FORMULAS = {
    CONTROL: 'reference87_once(.75*(.5*old_raw+.5*original_geo)+.25*old_unit)',
    CANDIDATE: 'reference87_once(.75*(.5*old_raw+.5*xgboost_geo)+.25*old_unit)',
}
V3_SHA = '52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
require, read, sha, rel = common.require, common.read, common.sha256, common.rel
locked, locked_write = common.read_locked, common.locked_write


def now():
    return evaluation.now()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def array_sha(values, dtype='<f8'):
    return hashlib.sha256(np.asarray(values, dtype=dtype).tobytes(order='C')).hexdigest()


def runtime():
    return {'python': platform.python_version(), **{n: importlib.metadata.version(n) for n in
            ['xgboost', 'numpy', 'pandas', 'scipy', 'catboost', 'lightgbm']}}


def sources():
    pending, found = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in found:
            continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = ([a.name for a in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                candidate = ROOT / 'src' / (name.split('.')[0] + '.py')
                if candidate.is_file() and candidate.resolve() not in found:
                    pending.append(candidate.resolve())
    return {rel(p): sha(p) for p in sorted(found)}


def stage_dependencies(stage):
    folder = common.baseline_folder(stage)
    paths = [folder / n for n in ['catboost_raw.npy', 'lightgbm_l1_unit63.npy',
             'reference_v2.pkl', 'fit_indices.npy', 'check_indices.npy']]
    geo_folder = common.OUT / 'models' / stage / 'geography_catboost'
    paths += [geo_folder / n for n in ['predictions.npy', 'fit_indices.npy', 'query_indices.npy']]
    paths += [common.OUT / 'evaluation' / stage / n for n in
              ['results.json', 'results.sha256', 'geography_blend.npy']]
    manifest_path = common.OUT / 'geography' / stage / 'manifest.json'
    manifest = locked(manifest_path)
    paths += [manifest_path, manifest_path.with_suffix('.sha256')]
    paths += [ROOT / v['path'] for v in manifest['features'].values()]
    paths += [ROOT / p for p in {**manifest['source_sha256'], **manifest['index_files_sha256']}]
    return {rel(p): sha(p) for p in sorted(set(paths))}


def plan_create():
    require(not (OUT / 'candidate_plan.json').exists(), 'Plan already exists')
    preflight = locked(OUT / 'preflight.json')
    require(preflight['checks']['status'] == 'passed' and preflight['source_sha256'] == sources()
            and preflight['runtime'] == runtime(), 'Formal helper preflight/source differs')
    old = common.plan()
    snapshot_path = ROOT / 'runs/v3/snapshot_manifest.json'
    require(sha(snapshot_path) == SNAPSHOT_SHA, 'Released v3 snapshot changed')
    snapshot = locked(snapshot_path)
    known = {p: v['original_sha256'] for p, v in snapshot['copied_files'].items()}
    for v in snapshot['external_dependencies']:
        require(v['original_path'] not in known or known[v['original_path']] == v['sha256'],
                'Conflicting archived identity')
        known[v['original_path']] = v['sha256']
    require(known.get(rel(common.OUT / 'candidate_plan.json')) == sha(common.OUT / 'candidate_plan.json'),
            'Original plan not identical to released v3')
    stages = {s: stage_dependencies(s) for s in STAGES}
    for mapping in stages.values():
        require(all(known.get(p) == digest for p, digest in mapping.items()),
                'Stage input not identical to released v3')
    research_root = ROOT / 'outputs/xgboost_research/feasibility_receipt_v2'
    proof = locked(research_root / 'feasibility.json')
    for field in ['source_sha256', 'input_sha256', 'artifact_sha256']:
        common.verify_hashes(proof[field])
    require(proof['status'] == 'synthetic_GPU_configuration_and_replay_verified'
            and proof['fits_started'] == proof['fits_completed'] == proof['fully_verified_models'] == 1
            and proof['rounds'] == 2,
            'Single artificial GPU compatibility test did not pass')
    require(proof['passed_parameters'] == component.parameters(), 'Probe and formal parameters differ')
    require(proof['prediction_replay']['raw_bitwise_equal'] is True
            and proof['prediction_replay']['TWD_bitwise_equal'] is True
            and proof['prediction_replay']['rows'] == 32,
            'Artificial model persistence did not pass')
    dependency = dict(old['data_sha256'])
    paths = [ROOT / 'outputs/submission_v3.csv', snapshot_path, snapshot_path.with_suffix('.sha256'),
             common.OUT / 'candidate_plan.json', common.OUT / 'candidate_plan.sha256',
             OUT / 'preflight.json', OUT / 'preflight.sha256',
             ROOT / 'reports/round32_plan.md', ROOT / 'reports/xgboost_feasibility_research.md',
             ROOT / 'outputs/xgboost_research/installation.json',
             ROOT / 'src/xgboost_feasibility.py', ROOT / 'src/xgboost_feasibility_receipt_v2.py',
             ROOT / 'outputs/xgboost_research/preflight.log',
             ROOT / 'outputs/xgboost_research/preflight_execution.json',
             ROOT / 'outputs/xgboost_research/feasibility/feasibility.json',
             ROOT / 'outputs/xgboost_research/feasibility/feasibility.sha256']
    paths += [p for p in research_root.rglob('*') if p.is_file()]
    dependency.update({rel(p): sha(p) for p in paths})
    require(dependency['outputs/submission_v3.csv'] == V3_SHA, 'v3 submission changed')
    require(runtime()['xgboost'] == '3.4.1', 'XGBoost version differs')
    value = {
        'round': 32, 'created_at_utc': now(), 'candidate': CANDIDATE, 'control': CONTROL,
        'source_sha256': sources(), 'dependency_sha256': dependency,
        'stage_dependency_sha256': stages, 'runtime': runtime(),
        'parameters': component.parameters(), 'num_boost_round': component.NUM_ROUNDS,
        'synthetic_fitted_native_config': read(research_root / 'fitted_config.json'),
        'native_comparison_excludes_execution_state': ['generic_param.rng_state'],
        'formulas': FORMULAS, 'thresholds': THRESHOLDS,
        'splits': old['splits'], 'split_contract': old['split_contract'],
        'target_group_weights': old['target_group_weights'], 'quarter_definition': old['quarter_definition'],
        'feature_count': 95, 'categorical_count': 23, 'fit_scope': 'all original fit rows',
        'target': 'y_TWD/1e6', 'inverse': 'max(z*1e6,0)', 'sample_weight': 'uniform',
        'family_comparison_not_single_parameter_causality': True,
        'vocabulary_fit_only': True, 'unseen_category': 'NaN',
        'query_labels_used_for_training': False, 'no_eval_set_or_early_stopping': True,
        'public_score_used_for_selection': False, 'historically_untouched': False,
        'external_property_answers_used': False, 'one_candidate_no_search': True,
        'final_fit_contract': old['final_contract'],
        'final_release_requires_all_three_existing_gates': True,
        'registry_sha256_at_plan': sha(ROOT / 'runs/versions.json'),
    }
    common.verify_hashes(dependency)
    require(value['source_sha256'] == preflight['source_sha256'] == sources(),
            'Sources changed during plan freeze')
    for original in [common.OUT / 'candidate_plan.json', common.OUT / 'candidate_plan.sha256']:
        require(sha(original) == known.get(rel(original)) == dependency[rel(original)],
                'Original R6 plan/receipt changed during freeze')
    locked_write(OUT / 'candidate_plan.json', value)
    print(json.dumps({'event': 'plan_frozen', 'sha256': sha(OUT / 'candidate_plan.json')}), flush=True)


def plan():
    p = locked(OUT / 'candidate_plan.json')
    require(p['round'] == 32 and p['candidate'] == CANDIDATE and p['control'] == CONTROL,
            'Candidate identity differs')
    require(p['source_sha256'] == sources() and p['runtime'] == runtime(), 'Source/runtime changed')
    require(p['parameters'] == component.parameters() and p['num_boost_round'] == component.NUM_ROUNDS
            and p['formulas'] == FORMULAS and p['thresholds'] == THRESHOLDS, 'Recipe changed')
    common.verify_hashes(p['dependency_sha256'])
    return p


def authorize(stage):
    require(stage in STAGES, 'Unknown stage')
    p = plan()
    common.verify_hashes(p['stage_dependency_sha256'][stage])
    if stage != 'development':
        frozen = locked(OUT / 'frozen_candidate.json')
        require(frozen['passed'] is True and frozen['plan_sha256'] == sha(OUT / 'candidate_plan.json'),
                'Development failed; later training prohibited')
        common.verify_hashes(frozen['evidence_sha256'])
    return p


def partition(stage, p):
    data, X, y, Xq, yq, fi, qi = common.partition(stage)
    spec = p['split_contract'][stage]
    require(len(X) == spec['fit_rows'] and len(Xq) == spec['query_rows'], 'Partition sizes changed')
    for name, indices in [('fit', fi), ('query', qi)]:
        require(array_sha(indices, '<i8') == spec[name + '_indices_content_sha256'], 'Partition changed')
    evaluation.verify_indices(common.baseline_folder(stage), fi, qi, 'check_indices.npy')
    evaluation.verify_indices(common.OUT / 'models' / stage / 'geography_catboost', fi, qi, 'query_indices.npy')
    return data, X, y, Xq, yq, fi, qi


def event(folder, name, **extra):
    with (folder / 'events.jsonl').open('a', encoding='utf-8') as stream:
        stream.write(json.dumps({'at': now(), 'event': name, **extra}, ensure_ascii=False, allow_nan=False) + '\n')
        stream.flush()
        os.fsync(stream.fileno())


def fit(stage):
    p = authorize(stage)
    folder = OUT / 'models' / stage
    folder.mkdir(parents=True, exist_ok=False)
    write_new(folder / 'run_claim.json', {'stage': stage, 'created_at': now(), 'plan_sha256': sha(OUT / 'candidate_plan.json')})
    try:
        data, X, y, Xq, yq, fi, qi = partition(stage, p)
        del yq
        geo, geoq, _ = load_geographic_partition(stage, fi, qi)
        before = {k: frame_sha(v) for k, v in {'fit_X': X, 'query_X': Xq, 'fit_geo': geo, 'query_geo': geoq}.items()}
        event(folder, 'fit_start', fit_rows=len(X), query_rows=len(Xq))
        print(json.dumps({'event': 'fit_start', 'stage': stage, 'fit_rows': len(X), 'query_rows': len(Xq)}), flush=True)
        started = time.monotonic()
        model, state, metadata = component.fit_model(X, y, geo, data['categorical_columns'])
        fitted = metadata['fitted_config']['learner']
        artificial = p['synthetic_fitted_native_config']['learner']
        for section in ['tree_train_param', 'gbtree_train_param']:
            require(fitted['gradient_booster'][section] == artificial['gradient_booster'][section],
                    'Native training settings differ from frozen artificial configuration: ' + section)
        # RNG state evolves with execution; seed and every actual setting stay fixed.
        stable_generic = lambda value: {k: v for k, v in value.items() if k != 'rng_state'}
        require(stable_generic(fitted['generic_param']) == stable_generic(artificial['generic_param']),
                'Native generic settings differ')
        event(folder, 'fit_complete', elapsed_seconds=time.monotonic() - started)
        z = component.raw_predict(model, state, Xq, geoq, data['categorical_columns'])
        pred = component.inverse_transform(z)
        model.save_model(str(folder / 'model.ubj'))
        write_new(folder / 'feature_state.json', state)
        reloaded = component.load_model(folder / 'model.ubj')
        replay = component.raw_predict(reloaded, read(folder / 'feature_state.json'), Xq, geoq, data['categorical_columns'])
        require(np.array_equal(z.view(np.uint64), replay.view(np.uint64)), 'Full-query raw replay differs')
        require(np.array_equal(pred.view(np.uint64), component.inverse_transform(replay).view(np.uint64)),
                'Full-query TWD replay differs')
        require(before == {k: frame_sha(v) for k, v in {'fit_X': X, 'query_X': Xq, 'fit_geo': geo, 'query_geo': geoq}.items()},
                'Input frames changed')
        for name, values in [('raw_z', z), ('predictions', pred), ('fit_indices', fi), ('query_indices', qi)]:
            np.save(folder / (name + '.npy'), values, allow_pickle=False)
        event(folder, 'save_reload_verified', query_rows=len(Xq), raw_negative_count=int((z < 0).sum()))
        metadata['component_source_sha256'] = metadata['source_sha256']
        metadata.update({
            'round': 32, 'stage': stage, 'created_at': now(), 'plan_sha256': sha(OUT / 'candidate_plan.json'),
            'source_sha256': p['source_sha256'], 'runtime': p['runtime'],
            'fit_rows': len(X), 'query_rows': len(Xq), 'fit_target_sha256': array_sha(y),
            'frame_sha256': before, 'fit_indices_sha256': array_sha(fi, '<i8'), 'query_indices_sha256': array_sha(qi, '<i8'),
            'reload_all_query_raw_and_TWD_bitwise_equal': True, 'raw_negative_count': int((z < 0).sum()),
            'shared_loader_deserialized_query_labels': True, 'query_labels_discarded_by_fit_caller': True,
            'query_labels_used_for_training': False, 'elapsed_seconds': time.monotonic() - started,
            'artifact_sha256': {rel(f): sha(f) for f in folder.iterdir() if f.is_file()},
        })
        locked_write(folder / 'metadata.json', metadata)
        print(json.dumps({'event': 'fit_completed', 'stage': stage, 'elapsed_seconds': metadata['elapsed_seconds']}), flush=True)
    except Exception as exc:
        event(folder, 'failed', error_type=type(exc).__name__, message=str(exc))
        raise


def decide(stage, metrics):
    base, candidate = metrics[CONTROL], metrics[CANDIDATE]
    improved = [q for q, v in base['forecast_quarters'].items() if v['rows'] and candidate['forecast_quarters'][q]['mae'] < v['mae']]
    checks = {'large_strata_deteriorate_at_most_5pct': evaluation.large_strata_ok(candidate, base)}
    if stage == 'development':
        checks.update(overall_improves_at_least_1pct=candidate['overall_mae'] <= (1 - THRESHOLDS['development_min_overall_improvement']) * base['overall_mae'],
                      weighted_improves_at_least_1pct=candidate['weighted_mae'] <= (1 - THRESHOLDS['development_min_weighted_improvement']) * base['weighted_mae'],
                      quarters_deteriorate_at_most_5pct=all(candidate['forecast_quarters'][q]['mae'] <= (1 + THRESHOLDS['development_max_quarter_deterioration']) * v['mae']
                      for q, v in base['forecast_quarters'].items() if v['rows']))
    else:
        checks.update(overall_within_limit=candidate['overall_mae'] <= (1 + THRESHOLDS[stage + '_max_overall_deterioration']) * base['overall_mae'],
                      weighted_within_limit=candidate['weighted_mae'] <= (1 + THRESHOLDS[stage + '_max_weighted_deterioration']) * base['weighted_mae'])
        if stage == 'confirmation':
            checks['at_least_two_quarters_improve'] = len(improved) >= THRESHOLDS['confirmation_min_improved_quarters']
    passed = all(checks.values())
    return {'passed': passed, 'checks': checks, 'improved_quarters': improved,
            'selected_candidate': CANDIDATE if passed else CONTROL}


def blend(old, geo, unit, reference, query):
    result = np.asarray(predict_reference(reference, query, .75 * (.5 * old + .5 * geo) + .25 * unit, variant='trend'), dtype='float64')
    require(result.shape == (len(query),) and np.isfinite(result).all() and (result >= 0).all(), 'Invalid composite TWD')
    return result


def evaluate(stage):
    p = authorize(stage)
    folder = OUT / 'evaluation' / stage
    require(not folder.exists(), 'Result already exists; do not re-score')
    model_folder = OUT / 'models' / stage
    metadata = locked(model_folder / 'metadata.json')
    require(metadata['plan_sha256'] == sha(OUT / 'candidate_plan.json') and metadata['source_sha256'] == p['source_sha256'],
            'Model metadata identity differs')
    common.verify_hashes(metadata['artifact_sha256'])
    require(metadata['reload_all_query_raw_and_TWD_bitwise_equal'] is True, 'Model persistence did not pass')
    data, X, y, Xq, yq, fi, qi = partition(stage, p)
    evaluation.verify_indices(model_folder, fi, qi, 'query_indices.npy')
    require(metadata['fit_target_sha256'] == array_sha(y), 'Training labels changed')
    weights = target_weights(data)
    require(weights == p['target_group_weights'], 'Scoring group weights changed')
    z = np.load(model_folder / 'raw_z.npy', allow_pickle=False)
    new = evaluation.vector(model_folder / 'predictions.npy', len(qi))
    require(np.array_equal(new.view(np.uint64), component.inverse_transform(z).view(np.uint64)), 'Saved inverse differs')
    reference, _ = common.load_reference(stage, X)
    baseline = common.baseline_folder(stage)
    old = evaluation.vector(baseline / 'catboost_raw.npy', len(qi))
    unit = evaluation.vector(baseline / 'lightgbm_l1_unit63.npy', len(qi))
    geo = evaluation.vector(common.OUT / 'models' / stage / 'geography_catboost/predictions.npy', len(qi))
    control = evaluation.vector(common.OUT / 'evaluation' / stage / 'geography_blend.npy', len(qi))
    require(np.array_equal(control.view(np.uint64), blend(old, geo, unit, reference, Xq).view(np.uint64)), 'v3 control replay differs')
    predictions = {CONTROL: control, CANDIDATE: blend(old, new, unit, reference, Xq)}
    metrics = {name: score(yq, pred, X, Xq, common.SPLITS[stage][0], weights) for name, pred in predictions.items()}
    require(metrics[CONTROL] == locked(common.OUT / 'evaluation' / stage / 'results.json')['metrics']['geography_blend'],
            'v3 control score differs from archive')
    folder.mkdir(parents=True, exist_ok=False)
    for name, values in predictions.items():
        np.save(folder / (name + '.npy'), values, allow_pickle=False)
    result = {
        'round': 32, 'stage': stage, 'created_at': now(), 'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'candidate': CANDIDATE, 'control': CONTROL, 'formulas': FORMULAS, 'thresholds': THRESHOLDS,
        'metrics': metrics, 'decision': decide(stage, metrics), 'target_group_weights': weights,
        'fit_rows': len(X), 'query_rows': len(Xq), 'source_sha256': p['source_sha256'],
        'input_artifact_sha256': {**p['stage_dependency_sha256'][stage], **metadata['artifact_sha256'],
            rel(model_folder / 'metadata.json'): sha(model_folder / 'metadata.json')},
        'output_artifact_sha256': {rel(f): sha(f) for f in folder.iterdir() if f.is_file()},
        'v3_control_all_rows_bitwise_replayed': True, 'component_separately_scored': False,
        'query_labels_used_for_training': False, 'public_score_used': False, 'historically_untouched': False,
    }
    locked_write(folder / 'results.json', result)
    if stage == 'development':
        locked_write(OUT / 'frozen_candidate.json', {
            'round': 32, 'plan_sha256': result['plan_sha256'], 'passed': result['decision']['passed'],
            'selected_candidate': result['decision']['selected_candidate'],
            'evidence_sha256': {rel(f): sha(f) for parent in [folder, model_folder] for f in parent.iterdir() if f.is_file()},
        })
    print(json.dumps({'stage': stage, 'metrics': metrics, 'decision': result['decision']}, ensure_ascii=False), flush=True)
    return result


def finalize():
    p = plan()
    development = locked(OUT / 'evaluation/development/results.json')
    results = {'development': development}
    if development['decision']['passed']:
        results.update({s: locked(OUT / 'evaluation' / s / 'results.json') for s in STAGES if s != 'development'})
    for stage, value in results.items():
        require(value['plan_sha256'] == sha(OUT / 'candidate_plan.json') and value['source_sha256'] == p['source_sha256']
                and value['decision'] == decide(stage, value['metrics']), 'Saved result contract differs')
        common.verify_hashes(value['input_artifact_sha256'])
        common.verify_hashes(value['output_artifact_sha256'])
    accepted = len(results) == 3 and all(v['decision']['passed'] for v in results.values())
    value = {'round': 32, 'created_at': now(), 'plan_sha256': sha(OUT / 'candidate_plan.json'),
             'accepted': accepted, 'selected_candidate': CANDIDATE if accepted else CONTROL,
             'evidence_sha256': {rel(OUT / 'evaluation' / s / 'results.json'): sha(OUT / 'evaluation' / s / 'results.json') for s in results},
             'decisions': {s: v['decision'] for s, v in results.items()},
             'final_training_allowed': accepted, 'new_CSV_created': False, 'registry_modified': False}
    locked_write(OUT / 'accepted_candidate.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['plan', 'fit', 'evaluate', 'finalize'])
    parser.add_argument('--stage', choices=STAGES)
    args = parser.parse_args()
    if args.action == 'plan':
        plan_create()
    elif args.action == 'finalize':
        finalize()
    else:
        require(args.stage is not None, '--stage is required')
        (fit if args.action == 'fit' else evaluate)(args.stage)
