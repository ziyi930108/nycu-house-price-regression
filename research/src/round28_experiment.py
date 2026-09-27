"""R28: one saved-prediction ablation; no fitting or native-model inference.
Replace the old-raw slot by the original R6 geographic prediction. Both complete
blends use original87/reference once. This does not score a standalone geo model.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import round6_common as c6
import round6_evaluate as e6
from round2_neighborhood import predict_reference
from round3_evaluate import score, target_weights, runtime_versions

ROOT = c6.ROOT
OUT = ROOT / 'outputs/round28'
STAGES = tuple(c6.SPLITS)
CONTROL = 'v3_control'
CANDIDATE = 'geography_replacement_blend'
SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA = '52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA = 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
FORMULAS = {
    CONTROL: '.75*(.5*old_raw_cat+.5*original_R6_geo_raw_cat)+.25*old_unit_lgb; then original87 v2 trend once',
    CANDIDATE: '.75*original_R6_geo_raw_cat+.25*old_unit_lgb; then original87 v2 trend once',
}
THRESHOLDS = dict(e6.THRESHOLDS)
require, read, locked, locked_write, sha, rel = (
    c6.require, c6.read, c6.read_locked, c6.locked_write, c6.sha256, c6.rel)

def now(): return datetime.now(timezone.utc).isoformat()
def value_sha(a): return hashlib.sha256(np.asarray(a, dtype='<i8').tobytes()).hexdigest()

def sources():
    pending, found = [Path(__file__).resolve()], set()
    while pending:
        path = pending.pop()
        if path in found: continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = ([n.name for n in node.names] if isinstance(node, ast.Import) else
                     [node.module] if isinstance(node, ast.ImportFrom) and node.module else [])
            for name in names:
                local = ROOT / 'src' / (name.split('.')[0] + '.py')
                if local.is_file() and local not in found: pending.append(local)
    return {rel(p): sha(p) for p in sorted(found)}

def snapshot_known():
    path = ROOT / 'runs/v3/snapshot_manifest.json'
    require(sha(path) == SNAPSHOT_SHA, 'Released v3 snapshot differs')
    snapshot = locked(path)
    known = {k: v['original_sha256'] for k, v in snapshot['copied_files'].items()}
    for item in snapshot['external_dependencies']:
        name, digest = item['original_path'], item['sha256']
        require(name not in known or known[name] == digest, 'Conflicting v3 identity')
        known[name] = digest
    return known

def stage_paths(stage):
    require(stage in STAGES, 'Only historical backtests are implemented')
    base = c6.baseline_folder(stage)
    geo = c6.OUT / 'models' / stage / 'geography_catboost'
    evaluation = c6.OUT / 'evaluation' / stage
    paths = [base / n for n in ('catboost_raw.npy', 'lightgbm_l1_unit63.npy',
                               'reference_v2.pkl', 'fit_indices.npy', 'check_indices.npy')]
    paths += [geo / n for n in ('metadata.json', 'predictions.npy', 'fit_indices.npy', 'query_indices.npy')]
    paths += [evaluation / n for n in ('results.json', 'results.sha256', 'geography_blend.npy')]
    # c6.load_reference also reads these saved provenance records.
    paths += ([base / 'v2_control.json'] if stage == 'auxiliary' else
              [ROOT / 'runs/round3_experiments/snapshot_manifest.json'])
    return sorted(set(paths))

def require_no_staging():
    require(not (OUT / 'models').exists(), 'R28 must not contain new models')
    folder = OUT / 'evaluation'
    if folder.exists():
        require(all(p.is_dir() and p.name in STAGES for p in folder.iterdir()),
                'Partial/unexpected evaluation artifact requires review')

def plan_create():
    require_no_staging()
    require(not (OUT / 'candidate_plan.json').exists() and
            not (OUT / 'candidate_plan.sha256').exists(), 'Plan already exists or is partial')
    known = snapshot_known()
    old_path = c6.OUT / 'candidate_plan.json'
    require(known.get(rel(old_path)) == sha(old_path), 'R6 plan is not released-v3 authority')
    old = c6.plan()
    captured_sources, captured_runtime = sources(), runtime_versions()
    preflight_path = OUT / 'audit/preflight.json'
    preflight = locked(preflight_path)
    require(preflight['round'] == 28 and preflight['all_checks_passed'] is True and
            preflight['real_model_training_or_scoring'] is False and
            preflight['source_sha256'] == captured_sources and preflight['runtime'] == captured_runtime,
            'Current source/runtime lack successful synthetic preflight')
    dependencies = dict(old['data_sha256'])
    required = [old_path, old_path.with_suffix('.sha256'),
                ROOT / 'runs/v3/snapshot_manifest.json', ROOT / 'runs/v3/snapshot_manifest.sha256',
                ROOT / 'outputs/submission_v2.csv', ROOT / 'outputs/submission_v3.csv',
                ROOT / 'reports/round28_plan.md', preflight_path, preflight_path.with_suffix('.sha256')]
    for path in required: dependencies[rel(path)] = sha(path)
    require(dependencies['outputs/submission_v2.csv'] == V2_SHA and
            dependencies['outputs/submission_v3.csv'] == V3_SHA, 'Protected submissions differ')
    stages = {}
    for stage in STAGES:
        stages[stage] = {rel(p): sha(p) for p in stage_paths(stage)}
        for name, digest in stages[stage].items():
            require(known.get(name) == digest, 'Saved stage input differs from released v3: ' + name)
    value = {
        'round': 28, 'created_at_utc': now(), 'candidate': CANDIDATE, 'control': CONTROL,
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'mode': 'fixed_saved_prediction_ablation',
        'source_sha256': captured_sources, 'dependency_sha256': dependencies,
        'stage_dependency_sha256': stages, 'runtime': captured_runtime,
        'old_plan_sha256': sha(old_path), 'v3_snapshot_sha256': SNAPSHOT_SHA,
        'splits': old['splits'], 'split_contract': old['split_contract'],
        'target_group_weights': old['target_group_weights'], 'quarter_definition': old['quarter_definition'],
        'original_geo_component_parameters': old['components']['geography_catboost']['parameters'],
        'reference_feature_count': 87, 'reused_geo_feature_count': 95, 'reused_geo_categorical_count': 23,
        'single_development_candidate': True, 'new_model_training': False,
        'native_model_replay_performed_here': False, 'tuning_or_weight_search': False,
        'candidate_component_separately_scored': False, 'reference_refitted': False,
        'query_labels_used_for_training': False, 'query_labels_used_for_saved_composition_scoring': True,
        'historically_untouched': False, 'external_property_answers_used': False,
        'public_score_used_for_selection': False, 'final_release_implemented': False,
        'registry_sha256_at_plan': sha(ROOT / 'runs/versions.json'),
    }
    c6.verify_hashes(dependencies)
    for mapping in stages.values(): c6.verify_hashes(mapping)
    require(captured_sources == sources() and captured_runtime == runtime_versions(), 'Source/runtime changed')
    require(known.get(rel(old_path)) == sha(old_path) == value['old_plan_sha256'] == dependencies[rel(old_path)],
            'R6 plan authority changed while freezing')
    old_receipt = old_path.with_suffix('.sha256')
    if rel(old_receipt) in known:
        require(known[rel(old_receipt)] == sha(old_receipt) == dependencies[rel(old_receipt)],
                'R6 plan receipt authority changed while freezing')
    locked_write(OUT / 'candidate_plan.json', value)
    print(json.dumps({'status': 'plan_frozen', 'plan_sha256': sha(OUT / 'candidate_plan.json')}), flush=True)
    return value

def plan():
    value = locked(OUT / 'candidate_plan.json')
    require(value['round'] == 28 and value['candidate'] == CANDIDATE and value['control'] == CONTROL,
            'Wrong R28 plan')
    require(value['formulas'] == FORMULAS and value['thresholds'] == THRESHOLDS and
            value['mode'] == 'fixed_saved_prediction_ablation', 'Fixed experiment changed')
    require(value['source_sha256'] == sources() and value['runtime'] == runtime_versions(), 'Source/runtime differs')
    require(value['splits'] == {k: list(v) for k, v in c6.SPLITS.items()}, 'Split definitions differ')
    require(value['new_model_training'] is False and value['tuning_or_weight_search'] is False and
            value['candidate_component_separately_scored'] is False and value['reference_refitted'] is False and
            value['final_release_implemented'] is False, 'Scope changed')
    require(value['old_plan_sha256'] == sha(c6.OUT / 'candidate_plan.json') and
            value['v3_snapshot_sha256'] == SNAPSHOT_SHA, 'Prior authority differs')
    c6.verify_hashes(value['dependency_sha256'])
    require(value['registry_sha256_at_plan'] == sha(ROOT / 'runs/versions.json'), 'Registry changed during experiment')
    return value

def vector(path, count):
    a = np.load(path, allow_pickle=False)
    require(a.dtype == np.dtype('float64') and a.shape == (count,) and
            np.isfinite(a).all() and (a >= 0).all(), 'Invalid saved TWD vector: ' + str(path))
    return a

def verify_indices(path, expected):
    actual = np.load(path, allow_pickle=False)
    require(actual.dtype == np.dtype('int64') and actual.ndim == 1 and
            np.array_equal(actual, expected), 'Saved index/order differs: ' + str(path))

def compose(old, geo, unit, reference, Xq, apply_reference=predict_reference):
    for values in (old, geo, unit):
        a = np.asarray(values)
        require(a.shape == (len(Xq),) and np.isfinite(a).all() and (a >= 0).all(), 'Invalid composition input')
    bases = {CONTROL: .75 * (.5 * old + .5 * geo) + .25 * unit,
             CANDIDATE: .75 * geo + .25 * unit}
    predictions = {}
    for name, base in bases.items():
        pred = np.asarray(apply_reference(reference, Xq, base, variant='trend'), dtype='float64')
        require(pred.shape == (len(Xq),) and np.isfinite(pred).all() and (pred >= 0).all(), 'Invalid composition output')
        predictions[name] = pred
    return predictions

def decide(stage, metrics):
    """Reuse R6 gates exactly, mapping names only; no independently new rule."""
    require(stage in STAGES and set(metrics) == {CONTROL, CANDIDATE}, 'Unexpected candidate/stage')
    mapped = {e6.CONTROL: metrics[CONTROL], 'geography_blend': metrics[CANDIDATE]}
    if stage == 'development':
        original = e6.development_selection(mapped)
        record = original['candidate_checks']['geography_blend']
        passed, checks = record['eligible'], record['checks']
        improved = [q for q, item in metrics[CONTROL]['forecast_quarters'].items()
                    if item['rows'] and metrics[CANDIDATE]['forecast_quarters'][q]['mae'] < item['mae']]
    else:
        original = e6.stage_decision(stage, mapped, 'geography_blend')
        passed, checks, improved = original['passed'], original['checks'], original['improved_quarters']
    return {'passed': passed, 'selected_candidate': CANDIDATE if passed else CONTROL,
            'checks': checks, 'improved_quarters': improved, 'candidate_reselection_performed': False}

def read_result(stage, specification):
    require(stage in STAGES, 'Unknown saved stage')
    folder = OUT / 'evaluation' / stage
    result = locked(folder / 'results.json')
    require(result['round'] == 28 and result['stage'] == stage and
            result['plan_sha256'] == sha(OUT / 'candidate_plan.json'), 'Result identity differs')
    require(result['source_sha256'] == specification['source_sha256'] and
            result['input_artifact_sha256'] == specification['stage_dependency_sha256'][stage] and
            result['formulas'] == FORMULAS and result['thresholds'] == THRESHOLDS and
            result['target_group_weights'] == specification['target_group_weights'], 'Saved result protocol differs')
    expected_freeze = None if stage == 'development' else sha(OUT / 'frozen_candidate.json')
    require(result['frozen_candidate_sha256'] == expected_freeze, 'Result freeze differs')
    require(result['decision'] == decide(stage, result['metrics']), 'Saved gate decision differs')
    contract = specification['split_contract'][stage]
    for name in ('fit_rows', 'query_rows', 'fit_indices_content_sha256', 'query_indices_content_sha256'):
        require(result[name] == contract[name], 'Result sample alignment differs')
    files = {CONTROL + '.npy', CANDIDATE + '.npy', 'fit_indices.npy', 'query_indices.npy'}
    require(set(result['output_artifact_sha256']) == {rel(folder / name) for name in files} and
            {p.name for p in folder.iterdir()} == files | {'results.json', 'results.sha256'}, 'Incomplete/unexpected result files')
    require(result['v3_control_bitwise_equal'] is True and result['new_model_training'] is False and
            result['native_model_replay_performed_here'] is False, 'Incorrect replay/training claims')
    c6.verify_hashes(result['input_artifact_sha256']); c6.verify_hashes(result['output_artifact_sha256'])
    return result

def checked_freeze(specification):
    frozen = locked(OUT / 'frozen_candidate.json')
    development = read_result('development', specification)
    require(frozen['round'] == 28 and frozen['candidate'] == CANDIDATE and
            frozen['plan_sha256'] == sha(OUT / 'candidate_plan.json') and
            frozen['result_sha256'] == sha(OUT / 'evaluation/development/results.json'), 'Freeze identity differs')
    require(frozen['decision'] == development['decision'] and
            frozen['selected_candidate'] == development['decision']['selected_candidate'] and
            frozen['passed'] == frozen['later_stages_allowed'] == development['decision']['passed'], 'Freeze decision differs')
    return frozen

def authorize(stage):
    require(stage in STAGES, 'Final release/new fitting is not implemented')
    specification = plan()
    c6.verify_hashes(specification['stage_dependency_sha256'][stage])
    if stage != 'development':
        frozen = checked_freeze(specification)
        require(frozen['passed'] is True and frozen['selected_candidate'] == CANDIDATE,
                'Current-round development did not pass; later scoring prohibited')
    return specification

def freeze_development(specification):
    result = read_result('development', specification)
    locked_write(OUT / 'frozen_candidate.json', {
        'round': 28, 'created_at_utc': now(), 'candidate': CANDIDATE,
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'result_sha256': sha(OUT / 'evaluation/development/results.json'),
        'decision': result['decision'], 'selected_candidate': result['decision']['selected_candidate'],
        'passed': result['decision']['passed'], 'later_stages_allowed': result['decision']['passed'],
    })

def evaluate(stage):
    require_no_staging()
    specification = authorize(stage)
    destination = OUT / 'evaluation' / stage
    require(not destination.exists(), 'Stage already exists; never rescore/overwrite')
    if stage == 'development':
        require(not (OUT / 'frozen_candidate.json').exists(), 'Development already frozen')
    data, X, y, Xq, yq, fi, qi = c6.partition(stage)
    del y  # Shared loader also reads fit targets; no training/statistics use here.
    contract = specification['split_contract'][stage]
    require(len(X.columns) == len(Xq.columns) == 87 and list(X.columns) == list(Xq.columns), 'Reference needs original87')
    for label, indices in (('fit', fi), ('query', qi)):
        require(len(indices) == contract[label + '_rows'] and
                value_sha(indices) == contract[label + '_indices_content_sha256'], 'Current partition differs')
    base = c6.baseline_folder(stage); geo = c6.OUT / 'models' / stage / 'geography_catboost'
    for folder, query_file in ((base, 'check_indices.npy'), (geo, 'query_indices.npy')):
        verify_indices(folder / 'fit_indices.npy', fi); verify_indices(folder / query_file, qi)
    meta = read(geo / 'metadata.json')
    require(meta['prediction_sha256'] == sha(geo / 'predictions.npy') and meta['prediction_units'] == 'TWD' and
            meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23 and
            meta['parameters'] == specification['original_geo_component_parameters'], 'Original geo prediction metadata differs')
    weights = target_weights(data)
    require(weights == specification['target_group_weights'], 'Fixed test-X score weights differ')
    reference, _ = c6.load_reference(stage, X)
    old = vector(base / 'catboost_raw.npy', len(qi))
    unit = vector(base / 'lightgbm_l1_unit63.npy', len(qi))
    geographic = vector(geo / 'predictions.npy', len(qi))
    original = vector(c6.OUT / 'evaluation' / stage / 'geography_blend.npy', len(qi))
    predictions = compose(old, geographic, unit, reference, Xq)
    require(np.array_equal(original.view('uint64'), predictions[CONTROL].view('uint64')),
            'Original v3 control fails exact all-query composition replay')
    metrics = {name: score(yq, pred, X, Xq, c6.SPLITS[stage][0], weights) for name, pred in predictions.items()}
    previous = locked(c6.OUT / 'evaluation' / stage / 'results.json')
    require(metrics[CONTROL] == previous['metrics']['geography_blend'], 'Original v3 score replay differs')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + stage + '-' + uuid.uuid4().hex); staging.mkdir()
    for name, values in predictions.items(): np.save(staging / (name + '.npy'), values, allow_pickle=False)
    np.save(staging / 'fit_indices.npy', np.asarray(fi, dtype='int64'), allow_pickle=False)
    np.save(staging / 'query_indices.npy', np.asarray(qi, dtype='int64'), allow_pickle=False)
    result = {
        'round': 28, 'stage': stage, 'created_at_utc': now(), 'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'frozen_candidate_sha256': None if stage == 'development' else sha(OUT / 'frozen_candidate.json'),
        'source_sha256': specification['source_sha256'],
        'input_artifact_sha256': specification['stage_dependency_sha256'][stage],
        'output_artifact_sha256': {rel(destination / p.name): sha(p) for p in staging.iterdir()},
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'target_group_weights': weights,
        'metrics': metrics, 'decision': decide(stage, metrics),
        'fit_rows': len(fi), 'query_rows': len(qi), 'fit_indices_content_sha256': value_sha(fi),
        'query_indices_content_sha256': value_sha(qi), 'v3_control_bitwise_equal': True,
        'verification_level': 'saved_prediction_composition_and_existing_score_replay',
        'new_model_training': False, 'native_model_replay_performed_here': False,
        'reference_refitted': False, 'candidate_component_separately_scored': False,
        'query_labels_used_for_training': False, 'query_labels_used_for_scoring': True,
        'historically_untouched': False, 'public_score_used': False, 'private_score_known': False,
    }
    locked_write(staging / 'results.json', result)
    authorize(stage)
    require(staging.resolve().parent == destination.resolve().parent and
            staging.resolve().is_relative_to(OUT.resolve()) and not destination.exists(), 'Unsafe publication')
    staging.rename(destination)
    if stage == 'development': freeze_development(specification)
    print(json.dumps({'stage': stage, 'decision': result['decision'], 'metrics': metrics}, ensure_ascii=False), flush=True)
    return result

def finalize():
    require_no_staging(); specification = plan(); frozen = checked_freeze(specification)
    results = {'development': read_result('development', specification)}
    statuses = {'development': 'price_evaluated'}
    for stage in ('confirmation', 'auxiliary'):
        if frozen['passed']:
            authorize(stage); results[stage] = read_result(stage, specification); statuses[stage] = 'price_evaluated'
        else:
            require(not (OUT / 'evaluation' / stage).exists(), 'Unauthorized later-stage artifacts')
            statuses[stage] = 'not_executed_after_development_rejection'
    accepted = len(results) == len(STAGES) and all(r['decision']['passed'] for r in results.values())
    reason = ('all_fixed_backtests_passed' if accepted else 'development_price_gates_failed'
              if not frozen['passed'] else 'later_backtest_gates_failed')
    result = {
        'round': 28, 'created_at_utc': now(), 'candidate': CANDIDATE,
        'plan_sha256': sha(OUT / 'candidate_plan.json'), 'frozen_candidate_sha256': sha(OUT / 'frozen_candidate.json'),
        'accepted': accepted, 'selected_candidate': frozen['selected_candidate'],
        'deployment_decision': 'backtest_accepted_pending_final_release' if accepted else 'keep_v3',
        'stopping_reason': reason, 'stage_status': statuses,
        'stage_result_sha256': {s: sha(OUT / 'evaluation' / s / 'results.json') for s in results},
        'unevaluated_stages': [s for s in STAGES if s not in results], 'source_sha256': specification['source_sha256'],
        'no_reselection_or_parameter_change': True, 'new_model_training': False, 'new_submission_created': False,
        'currently_recommended_submission': 'outputs/submission_v3.csv', 'historically_untouched': False,
        'private_score_known': False,
    }
    locked_write(OUT / 'accepted_candidate.json', result)
    print(json.dumps(result, ensure_ascii=False), flush=True)
    return result

def self_check(write_output=True):
    """Small new-composition/name-mapping checks; no models, data, prices or full inherited suite."""
    checks = []
    def ok(name, condition): require(bool(condition), 'Synthetic: ' + name); checks.append(name)
    def reject(name, fn):
        try: fn()
        except (RuntimeError, ValueError): checks.append(name); return
        raise RuntimeError('Synthetic rejection failed: ' + name)
    old, geo, unit = np.array([1., 4.]), np.array([3., 8.]), np.array([2., 6.])
    Xq = object(), object(); calls = []
    def reference(_, query, base, variant):
        calls.append((query, base.copy(), variant)); return base + 1.
    pred = compose(old, geo, unit, None, Xq, reference)
    ok('exact_control_formula', np.array_equal(pred[CONTROL], .75 * (.5 * old + .5 * geo) + .25 * unit + 1))
    ok('exact_single_candidate_formula', np.array_equal(pred[CANDIDATE], .75 * geo + .25 * unit + 1))
    ok('one_reference_call_per_complete_blend_original_query', len(calls) == 2 and all(q is Xq and v == 'trend' for q, b, v in calls))
    changed = compose(old * 10, geo, unit, None, Xq, reference)
    ok('candidate_does_not_use_oldraw', np.array_equal(changed[CANDIDATE], pred[CANDIDATE]))
    reject('nan_component_rejected', lambda: compose(old, geo * np.nan, unit, None, Xq, reference))
    reject('misaligned_component_rejected', lambda: compose(old[:1], geo, unit, None, Xq, reference))
    def metric(value):
        return {'overall_mae': value, 'weighted_mae': value, 'strata': {'new': {'rows': 501, 'mae': value}},
                'forecast_quarters': {q: {'rows': 10, 'mae': value} for q in ('Q1', 'Q2', 'Q3', 'Q4')}}
    metrics = {CONTROL: metric(100.), CANDIDATE: metric(98.)}
    ok('development_name_mapping_pass', decide('development', metrics)['passed'])
    ok('confirmation_name_mapping_pass', decide('confirmation', metrics)['passed'])
    ok('auxiliary_name_mapping_pass', decide('auxiliary', metrics)['passed'])
    ok('development_name_mapping_reject', not decide('development', {CONTROL: metric(100.), CANDIDATE: metric(100.)})['passed'])
    reject('undeclared_candidate_rejected', lambda: decide('development', {CONTROL: metric(100.), 'other': metric(98.)}))
    reject('final_stage_rejected', lambda: decide('final', metrics))
    value = {'round': 28, 'created_at_utc': now(), 'all_checks_passed': True,
             'real_model_training_or_scoring': False, 'checks': checks, 'check_count': len(checks),
             'source_sha256': sources(), 'runtime': runtime_versions()}
    if write_output: locked_write(OUT / 'audit/preflight.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)
    return value

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    check = commands.add_parser('self-check'); check.add_argument('--no-write', action='store_true')
    commands.add_parser('plan'); commands.add_parser('finalize')
    commands.add_parser('evaluate').add_argument('--stage', choices=STAGES, required=True)
    args = parser.parse_args()
    if args.command == 'self-check': self_check(not args.no_write)
    elif args.command == 'plan': plan_create()
    elif args.command == 'evaluate': evaluate(args.stage)
    else: finalize()
