"""Archive one finalized R7 experiment without publishing a release or registry change.

Default writes a new runs/round7_experiments snapshot; --verify is read-only.
Only current-round artifacts/sources are copied. Older sources, raw/cache files,
all frozen prior-stage dependencies and maps are retained as exact-SHA externals.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, os, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round7'
DEST = ROOT / 'runs/round7_experiments'
MANIFEST, RECEIPT, NOTICE_FILE = 'snapshot_manifest.json', 'snapshot_manifest.sha256', 'SNAPSHOT_README.md'
CANDIDATE, CONTROL = 'geography_unit_replacement', 'v3_control'
STAGES = ('development', 'confirmation', 'auxiliary')
FORMULAS = {
    CONTROL: '.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once',
    CANDIDATE: '.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*geo_unit_lgb; then v2 trend once',
}
REQUIRED_REPORTS = ('README_round7.md', 'reports/optimization_round7_report.md',
    'reports/round7_results.json', 'reports/round7_work_log.md',
    'reports/round7_feature_audit.md', 'reports/round7_validation_plan.md')
SUFFIXES = {'.json', '.sha256', '.csv', '.npy', '.pkl', '.txt', '.md', '.py', '.png', '.svg', '.pdf', '.html', '.ipynb'}
OMIT_DIRS = {'__pycache__', 'logs', 'log', 'matplotlib_cache', 'notebook_runtime'}
NOTICE = '''# R7 finalized experiment snapshot

This is an experiment archive, not a new release or Kaggle submission. It
preserves all completed R7 model/evaluation stages, decisions, receipts, current
round source files, audits and reports. A failed development has one completed
stage; a passed development requires both later stages, whether accepted or
rejected. Accepted means the fixed historical backtest gates passed only.

Original spreadsheets, feature caches, all older sources, previous stage and
final-stage dependencies, prior releases and large geographic files remain
exact-SHA external_dependencies. Restore their exact paths and content to replay;
this snapshot alone is not self-contained. No older geographic tree is copied.

The manifest excludes itself and has its own SHA receipt. Original and copied
file hashes and sizes are recorded separately. Failed staging is retained.
The v3/v2 submissions, R6/v3 snapshot and versions registry are unchanged.
No new Public or Private result, new CSV, upload or release is claimed.
'''


def need(ok, message):
    if not ok: raise RuntimeError(message)


def safe(base, name):
    base = Path(base).resolve(); path = (base / Path(name)).resolve()
    need(path != base and path.is_relative_to(base), 'Path escapes workspace: ' + str(name))
    return path


def rel(path): return safe(ROOT, path).relative_to(ROOT.resolve()).as_posix()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encoded(value): return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')
def content_sha(value): return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False).encode()).hexdigest()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''): digest.update(block)
    return digest.hexdigest()


def verify_file(base, name, digest, size=None):
    path = safe(base, name)
    need(path.is_file(), 'Missing file: ' + str(path))
    need(size is None or path.stat().st_size == size, 'File size differs: ' + str(path))
    need(sha(path) == digest, 'File SHA differs: ' + str(path))
    return path


def registry():
    expected, reasons = {}, {}
    def add(name, digest=None, reason='declared dependency'):
        path = safe(ROOT, name); key = rel(path)
        need(path.is_file(), 'Missing dependency: ' + key)
        actual = expected[key] if key in expected else sha(path)
        need(digest is None or (isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) and digest == actual), 'Declared SHA differs: ' + key)
        expected[key] = actual; reasons.setdefault(key, set()).add(reason)
        return actual
    def mapping(value, reason):
        need(isinstance(value, dict), 'Expected SHA mapping: ' + reason)
        for name, digest in value.items(): add(name, digest, reason)
    def locked(name, reason):
        path = safe(ROOT, name); receipt = path.with_suffix('.sha256')
        digest = add(path, reason=reason); add(receipt, reason=reason + ' receipt')
        need(receipt.read_text(encoding='ascii').strip() == digest, 'Receipt differs: ' + rel(path))
        return read(path)
    return expected, reasons, add, mapping, locked


def require_false(value, keys, label):
    for key in keys: need(value[key] is False, label + ': unexpected ' + key)


def validate_decision(stage, decision):
    common = {'large_strata_deteriorate_at_most_5pct'}
    if stage == 'development':
        keys = common | {'overall_improves_at_least_1pct', 'weighted_improves_at_least_1pct', 'quarters_deteriorate_at_most_5pct'}
    else:
        keys = common | {'overall_within_limit', 'weighted_within_limit'}
        if stage == 'confirmation': keys.add('at_least_two_quarters_improve')
    need(set(decision['checks']) == keys and all(type(v) is bool for v in decision['checks'].values()), 'Decision check schema differs: ' + stage)
    need(type(decision['passed']) is bool and decision['passed'] == all(decision['checks'].values()), 'Saved pass/checks differ: ' + stage)
    need(decision['selected_candidate'] == (CANDIDATE if decision['passed'] else CONTROL), 'Saved selection differs: ' + stage)
    need(decision['candidate_reselection_performed'] is False, 'Candidate reselected')
    need(isinstance(decision['improved_quarters'], list) and len(set(decision['improved_quarters'])) == len(decision['improved_quarters']) and set(decision['improved_quarters']) <= {'Q1', 'Q2', 'Q3', 'Q4'}, 'Quarter decision schema differs')


def verify_indices(path, rows, digest):
    array = np.load(path, allow_pickle=False, mmap_mode='r')
    need(array.shape == (rows,) and array.dtype.kind == 'i' and array.dtype.itemsize == 8, 'Index schema differs: ' + str(path))
    need((array >= 0).all() and (len(array) < 2 or (np.diff(array) > 0).all()), 'Indices are not unique sorted row indices')
    need(hashlib.sha256(np.asarray(array, dtype='<i8').tobytes()).hexdigest() == digest, 'Index content differs: ' + str(path))
    return array


def verify_model(stage, plan, frozen_sha, add, locked):
    folder = OUT / 'models' / stage
    metadata = locked(folder / 'metadata.json', stage + ' model metadata')
    spec = plan['split_contract'][stage]
    artifacts = {'model.txt', 'feature_state.json', 'predictions.npy', 'fit_indices.npy', 'query_indices.npy'}
    need(set(metadata['artifact_sha256']) == artifacts, 'Model artifact inventory differs')
    need({p.name for p in folder.iterdir() if p.is_file()} == artifacts | {'metadata.json', 'metadata.sha256'}, 'Unexpected model files')
    for name, digest in metadata['artifact_sha256'].items(): add(folder / name, digest, stage + ' model artifact')
    need(metadata['round'] == 7 and metadata['stage'] == stage and metadata['component'] == 'geography_unit_lightgbm', 'Model identity differs')
    need(metadata['plan_sha256'] == sha(OUT / 'candidate_plan.json') and metadata['source_sha256'] == plan['source_sha256'], 'Model plan/source differs')
    need(metadata['parameters'] == plan['parameters'] and metadata['iterations'] == metadata['actual_iterations'] == metadata['actual_trees'] == 3999, 'Model parameters/iterations differ')
    need(metadata['feature_count'] == 95 and metadata['categorical_feature_count'] == 23, 'Model feature counts differ')
    need(metadata['outer_fit_rows'] == spec['fit_rows'] and metadata['query_rows'] == spec['query_rows'], 'Model row counts differ')
    need(metadata['frozen_candidate_sha256'] == (None if stage == 'development' else frozen_sha), 'Model freeze differs')
    need(metadata['reload_64_rows_verified'] is True and metadata['category_maps_fit_only'] is True and metadata['prediction_units'] == 'TWD', 'Model replay/encoding/units contract differs')
    require_false(metadata, ['fit_uses_eval_set', 'early_stopping_used', 'query_labels_used', 'external_property_answers_used', 'target_used_for_geographic_state', 'query_used_for_geographic_state'], 'Model')
    need(metadata['fit_target_sha256'] == metadata['fit_target_values_sha256'], 'Raw target identity fields differ')
    manifest_key = 'outputs/round6/geography/' + stage + '/manifest.json'
    need(metadata['geo_manifest_sha256'] == plan['stage_dependency_sha256'][stage][manifest_key], 'Model geography version differs')
    state = read(folder / 'feature_state.json')
    need(state['state_content_sha256'] == content_sha({k: v for k, v in state.items() if k != 'state_content_sha256'}) == metadata['feature_state_content_sha256'], 'Feature state content differs')
    need(state['source_sha256'] == plan['source_sha256']['src/round7_geo_unit.py'] and state['parameters'] == plan['parameters'], 'Feature state source/parameters differ')
    need(state['dependency_source_sha256'] == metadata['dependency_source_sha256'] and all(plan['source_sha256'].get(k) == v for k, v in state['dependency_source_sha256'].items()), 'Feature state dependencies differ')
    need(state['base_configuration_sha256'] == metadata['base_configuration_sha256'] == plan['dependency_sha256']['runs/v1/models/final_config.json'], 'Base configuration differs')
    need(state['feature_rules'] == metadata['geo_feature_rules'] and state['feature_rules']['iterations'] == 3999 and state['feature_rules']['unseen_category_code'] == -1, 'Feature rules differ')
    require_false(state, ['fit_state_uses_y', 'fit_state_uses_query'], 'Feature state')
    need(state['fit_rows'] == spec['fit_rows'] and state['fit_original_feature_values_sha256'] == metadata['fit_X_sha256'] == metadata['fit_original_feature_values_sha256'] and state['fit_geo_feature_values_sha256'] == metadata['fit_geo_sha256'] == metadata['fit_geo_feature_values_sha256'], 'Fit state/value identity differs')
    config = read(ROOT / 'runs/v1/models/final_config.json')
    need(state['original_columns'] == config['feature_columns'] and state['original_categorical_columns'] == config['categorical_columns'], 'Original feature/category schema differs')
    need(len(state['feature_columns']) == 95 and len(state['categorical_columns']) == 23 and state['all_feature_names'] == state['feature_columns'] == metadata['feature_columns'], 'Augmented schema differs')
    safe_names = {name: 'f%03d' % i for i, name in enumerate(state['feature_columns'])}
    need(state['safe_feature_names'] == safe_names and state['model_feature_columns'] == list(safe_names.values()) == metadata['model_feature_columns'], 'Model feature mapping differs')
    need(state['categorical_columns'] == metadata['categorical_columns'] and state['categorical_features'] == [safe_names[c] for c in state['categorical_columns']] == metadata['categorical_features'], 'Categorical model schema differs')
    need(set(state['category_maps']) == set(state['categorical_columns']), 'Category map inventory differs')
    for mapping in state['category_maps'].values():
        need(all(isinstance(k, str) and type(v) is int for k, v in mapping.items()) and sorted(mapping.values()) == list(range(len(mapping))), 'Invalid category codes')
    headers, trees = {}, []
    with (folder / 'model.txt').open(encoding='utf-8') as handle:
        for line in handle:
            if line.startswith('Tree='): trees.append(int(line.strip().split('=', 1)[1]))
            elif not trees and '=' in line:
                k, v = line.rstrip('\n').split('=', 1); headers[k] = v
    need(trees == list(range(3999)), 'Native saved tree inventory differs')
    need(headers.get('feature_names', '').split() == state['model_feature_columns'] and headers.get('max_feature_idx') == '94' and headers.get('num_tree_per_iteration') == '1' and headers.get('objective') == 'regression_l1', 'Native saved model schema/objective differs')
    indices = {}
    for role in ('fit', 'query'):
        indices[role] = verify_indices(folder / (role + '_indices.npy'), spec[role + '_rows'], spec[role + '_indices_content_sha256'])
    for name in plan['stage_dependency_sha256'][stage]:
        role = 'fit' if Path(name).name == 'fit_indices.npy' else 'query' if Path(name).name in {'query_indices.npy', 'check_indices.npy'} else None
        if role:
            need(np.array_equal(indices[role], np.load(ROOT / name, allow_pickle=False, mmap_mode='r')), 'Prior-stage row alignment differs: ' + name)
    prediction = np.load(folder / 'predictions.npy', allow_pickle=False, mmap_mode='r')
    need(prediction.shape == (spec['query_rows'],) and prediction.dtype == np.dtype('float64') and np.isfinite(prediction).all() and (prediction >= 0).all(), 'Saved model prediction schema differs')
    return metadata


def verify_prior(plan, add, locked):
    need(plan['v3_snapshot_sha256'] == plan['dependency_sha256']['runs/v3/snapshot_manifest.json'], 'Prior snapshot identity differs')
    v3 = locked('runs/v3/snapshot_manifest.json', 'preserved R6/v3 snapshot')
    need(v3['version'] == 'v3' and v3['experiment_round'] == 6 and v3['is_release'] is True, 'Wrong preserved snapshot')
    # The released snapshot also binds the large maps and old baseline models.
    # Keep those as externals, but verify their complete original SHA/size chain.
    for item in v3['external_dependencies']:
        path = safe(ROOT, item['original_path'])
        need(path.is_file() and path.stat().st_size == item['bytes'],
             'Prior v3 external dependency size differs: ' + item['original_path'])
        add(path, item['sha256'], 'preserved R6/v3 external dependency')
    for stage in (*STAGES, 'final'):
        name = 'outputs/round6/geography/' + stage + '/manifest.json'
        dependencies = plan['final_dependency_sha256'] if stage == 'final' else plan['stage_dependency_sha256'][stage]
        need(name in dependencies, 'Geography manifest is not bound by R7 plan: ' + stage)
        add(name, dependencies[name], 'planned ' + stage + ' geography manifest')
        geography = locked(name, 'preserved ' + stage + ' geography provenance')
        need(geography['stage'] == stage and geography['geography_provenance_chain']['verified'] is True,
             'Geography stage or provenance certificate differs: ' + stage)
        for field in ('source_sha256', 'index_files_sha256'):
            need(isinstance(geography[field], dict), 'Missing geographic SHA mapping: ' + field)
            for source_name, digest in geography[field].items():
                add(source_name, digest, stage + ' full geography ' + field)
    for name, item in v3['copied_files'].items():
        need(item['original_sha256'] == item['backup_sha256'] and item['original_bytes'] == item['backup_bytes'], 'Prior v3 copy proof differs')
        path = safe(ROOT / 'runs/v3', name)
        need(path.stat().st_size == item['backup_bytes'], 'Prior v3 copy size differs')
        add(path, item['backup_sha256'], 'preserved R6/v3 copied file')
    actual = {p.relative_to(ROOT / 'runs/v3').as_posix() for p in (ROOT / 'runs/v3').rglob('*') if p.is_file()}
    need(actual == set(v3['copied_files']) | {MANIFEST, RECEIPT, v3['dependency_notice'], 'REGISTRY_BEFORE.json'}, 'Prior v3 inventory differs')
    for name in (v3['dependency_notice'], 'REGISTRY_BEFORE.json'): add(ROOT / 'runs/v3' / name, reason='preserved v3 notice/context')
    before = v3['original_plan']['registry_before']
    need(sha(ROOT / 'runs/v3/REGISTRY_BEFORE.json') == before['sha256'], 'Prior v3 registry backup differs')
    tree = ast.parse((ROOT / 'runs/v3/src/archive_round6_release.py').read_text(encoding='utf-8-sig'))
    notice = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NOTICE' for t in n.targets)]
    need(len(notice) == 1 and (ROOT / 'runs/v3' / v3['dependency_notice']).read_bytes() == notice[0].encode(), 'Prior v3 notice differs')
    prior = v3['original_plan']['prior_archives_verified']
    add(prior['v2_manifest_path'], prior['v2_manifest_sha256'], 'preserved v2 manifest')
    v2 = read(ROOT / prior['v2_manifest_path']); need(v2['version'] == 'v2', 'Wrong preserved v2')
    for name, digest in v2['files_sha256'].items(): add(ROOT / 'runs/v2' / name, digest, 'preserved v2 snapshot file')
    add('runs/versions.json', plan['registry_sha256_at_plan'], 'registry must remain unchanged')
    need(read(ROOT / 'runs/versions.json')['recommended_version'] == 'v3', 'Prior recommendation changed')
    return {'v3_manifest_sha256':plan['v3_snapshot_sha256'], 'v3_verified_files':len(v3['copied_files']), 'v2_manifest_sha256':prior['v2_manifest_sha256'], 'v2_verified_files':len(v2['files_sha256']), 'registry_sha256':plan['registry_sha256_at_plan']}


def source_closure(paths):
    found, pending = set(), list(paths)
    while pending:
        path = safe(ROOT, pending.pop())
        if path in found: continue
        need(path.is_file(), 'Missing source: ' + str(path)); found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                local = ROOT / 'src' / (name.split('.')[0] + '.py')
                if local.is_file() and local not in found: pending.append(local)
    return found


def validate_contracts(add, mapping, locked):
    plan = locked(OUT / 'candidate_plan.json', 'frozen R7 plan')
    frozen = locked(OUT / 'frozen_candidate.json', 'frozen development decision')
    accepted = locked(OUT / 'accepted_candidate.json', 'final R7 decision')
    psha, fsha = sha(OUT / 'candidate_plan.json'), sha(OUT / 'frozen_candidate.json')
    need(plan['round'] == 7 and plan['candidate'] == CANDIDATE and plan['control'] == CONTROL, 'Wrong experiment identity')
    need(plan['formulas'] == FORMULAS and plan['feature_count'] == 95 and plan['categorical_count'] == 23, 'Fixed formula/features differ')
    need(plan['single_development_candidate'] is True, 'Unexpected candidate search')
    require_false(plan, ['tuning_or_weight_search', 'public_score_used_for_selection', 'historically_untouched', 'external_property_answers_used', 'fit_uses_eval_set', 'early_stopping_used'], 'Plan')
    need(set(plan['splits']) == set(plan['split_contract']) == set(plan['stage_dependency_sha256']) == set(STAGES), 'Planned stage inventory differs')
    actual_sources = {rel(p):sha(p) for p in source_closure([ROOT / 'src/round7_experiment.py'])}
    need(plan['source_sha256'] == actual_sources, 'Frozen experiment import closure changed')
    mapping(plan['source_sha256'], 'frozen R7 sources')
    mapping(plan['dependency_sha256'], 'frozen plan dependency')
    for stage, values in plan['stage_dependency_sha256'].items(): mapping(values, stage + ' old dependencies')
    mapping(plan['final_dependency_sha256'], 'old final-stage dependencies')
    old = locked('outputs/round6/candidate_plan.json', 'retained R6 plan')
    need(plan['old_plan_sha256'] == sha(ROOT / 'outputs/round6/candidate_plan.json'), 'Retained R6 plan identity differs')
    for key in ('thresholds', 'splits', 'split_contract', 'target_group_weights', 'quarter_definition', 'final_contract'):
        need(plan[key] == old[key], 'R6/R7 fixed contract differs: ' + key)
    cfg = read(ROOT / 'runs/v1/models/final_config.json')
    entries = [v for v in cfg['components'] if v['name'] == 'lightgbm_l1_unit63']
    need(len(entries) == 1 and entries[0]['iterations'] == 3999, 'Wrong base unit model')
    need(plan['parameters'] == {**entries[0]['parameters'], 'n_estimators':3999, 'n_jobs':4}, 'Full original LGB parameters differ')
    preflight = read(OUT / 'audit/preflight.json')
    need(preflight['all_checks_passed'] is True and preflight['source_sha256'] == plan['source_sha256'] and preflight['parameters'] == plan['parameters'] and preflight['real_model_training_or_scoring'] is False, 'Preflight contract differs')
    need(frozen['plan_sha256'] == psha and frozen['source_sha256'] == plan['source_sha256'], 'Freeze plan/source differs')
    for key in ('formulas', 'thresholds', 'parameters'): need(frozen[key] == plan[key], 'Frozen configuration differs: ' + key)
    need(type(frozen['passed']) is bool and frozen['later_stages_allowed'] == frozen['passed'], 'Frozen authorization differs')
    need(frozen['selected_candidate'] == (CANDIDATE if frozen['passed'] else CONTROL), 'Frozen candidate differs')
    mapping(frozen['development_artifact_sha256'], 'frozen development artifacts')
    completed = list(STAGES) if frozen['passed'] else ['development']
    for kind in ('models', 'evaluation'):
        parent = OUT / kind
        actual = {p.name for p in parent.iterdir() if p.is_dir() and not p.name.startswith('.')} if parent.exists() else set()
        need(actual == set(completed), 'Actual completed stage directories differ: ' + kind)
    results, metadata = {}, {}
    for stage in completed:
        folder = OUT / 'evaluation' / stage
        value = locked(folder / 'results.json', stage + ' saved evaluation')
        spec = plan['split_contract'][stage]
        need(value['round'] == 7 and value['stage'] == stage and value['plan_sha256'] == psha, 'Result identity differs')
        need(value['frozen_candidate_sha256'] == (None if stage == 'development' else fsha), 'Result freeze differs')
        need(value['source_sha256'] == plan['source_sha256'], 'Result source differs')
        for key in ('formulas', 'thresholds', 'target_group_weights'): need(value[key] == plan[key], 'Result fixed configuration differs: ' + key)
        need(value['evaluated_candidates'] == [CONTROL, CANDIDATE] and set(value['metrics']) == {CONTROL, CANDIDATE}, 'Unexpected scored candidate')
        need(value['fit_rows'] == spec['fit_rows'] and value['query_rows'] == spec['query_rows'], 'Result counts differ')
        need(value['v3_control_replayed_all_rows'] is True and value['new_model_replayed_all_rows'] is True, 'All-row replay proof absent')
        require_false(value, ['candidate_component_separately_scored', 'query_labels_used_for_training', 'public_score_used', 'historically_untouched', 'new_public_score_known', 'private_score_known'], 'Result')
        validate_decision(stage, value['decision'])
        mapping(value['input_artifact_sha256'], stage + ' evaluated inputs')
        mapping(value['output_artifact_sha256'], stage + ' evaluated outputs')
        need(value['output_artifact_sha256'] == {rel(folder / (name + '.npy')):sha(folder / (name + '.npy')) for name in (CONTROL, CANDIDATE)}, 'Evaluation prediction inventory differs')
        need({p.name for p in folder.iterdir() if p.is_file()} == {'results.json', 'results.sha256', CONTROL + '.npy', CANDIDATE + '.npy'}, 'Unexpected evaluation files')
        for name, digest in plan['stage_dependency_sha256'][stage].items(): need(value['input_artifact_sha256'].get(name) == digest, 'Evaluation omits/changes old stage dependency: ' + name)
        metadata[stage] = verify_model(stage, plan, fsha, add, locked)
        for path in (OUT / 'models' / stage).iterdir():
            if path.is_file(): need(value['input_artifact_sha256'].get(rel(path)) == sha(path), 'Evaluation omits/changes model artifact')
        for name in (CONTROL, CANDIDATE):
            array = np.load(folder / (name + '.npy'), allow_pickle=False, mmap_mode='r')
            need(array.shape == (spec['query_rows'],) and array.dtype == np.dtype('float64') and np.isfinite(array).all() and (array >= 0).all(), 'Evaluation vector schema differs')
        results[stage] = value
    development = results['development']
    need(frozen['result_sha256'] == sha(OUT / 'evaluation/development/results.json') and frozen['passed'] == development['decision']['passed'] and frozen['selected_candidate'] == development['decision']['selected_candidate'], 'Freeze differs from saved development')
    frozen_files = {rel(p):sha(p) for parent in (OUT / 'models/development', OUT / 'evaluation/development') for p in parent.iterdir() if p.is_file()}
    need(frozen['development_artifact_sha256'] == frozen_files, 'Frozen development inventory differs')
    outcome = len(completed) == 3 and all(v['decision']['passed'] for v in results.values())
    need(accepted['round'] == 7 and accepted['candidate'] == CANDIDATE and accepted['plan_sha256'] == psha and accepted['frozen_candidate_sha256'] == fsha, 'Final decision identity differs')
    need(type(accepted['accepted']) is bool and accepted['accepted'] == outcome, 'Final acceptance differs from completed saved decisions')
    need(accepted['deployment_decision'] == (CANDIDATE if outcome else 'keep_v3') and accepted['no_reselection_or_parameter_change'] is True, 'Final deployment rule differs')
    need(accepted['source_sha256'] == plan['source_sha256'], 'Final decision source differs')
    need(accepted['result_sha256'] == {stage:sha(OUT / 'evaluation' / stage / 'results.json') for stage in completed} and accepted['unevaluated_stages'] == [s for s in STAGES if s not in completed], 'Final completed/unexecuted stages differ')
    require_false(accepted, ['new_submission_created', 'private_score_known'], 'Final decision')
    return plan, frozen, accepted, results, metadata, completed


def validate_reports(plan, frozen, accepted, results, metadata, add, mapping):
    for name in REQUIRED_REPORTS: add(name, reason='required R7 report')
    report = read(ROOT / 'reports/round7_results.json')
    need(report['round'] == 7 and report['metric'] == 'MAE' and report['unit'] == 'TWD', 'Report identity/units differ')
    need(report['plan_sha256'] == sha(OUT / 'candidate_plan.json') and report['frozen_candidate_sha256'] == sha(OUT / 'frozen_candidate.json') and report['accepted_candidate_sha256'] == sha(OUT / 'accepted_candidate.json'), 'Report contracts differ')
    need(report['acceptance'] == accepted and report['acceptance_finalized'] is True and report['backtest_accepted'] == accepted['accepted'], 'Report final decision differs')
    need(report['selected_candidate'] == frozen['selected_candidate'], 'Report frozen selection differs')
    need(report['recommendation'] == 'v3' and report['recommended_submission'] == 'outputs/submission_v3.csv' and report['new_submission'] is None and report['new_release_verified'] is False, 'Experiment report falsely claims a new release')
    need(report['public_mae'] is None and report['private_mae'] is None and report['historically_untouched'] is False, 'Report claims unknown score or independent holdout')
    for key in ('formulas', 'thresholds', 'parameters'): need(report[key] == plan[key], 'Report rules differ: ' + key)
    need(set(report['stages']) == set(STAGES), 'Report stage inventory differs')
    for stage in STAGES:
        item = report['stages'][stage]; spec = plan['split_contract'][stage]
        need(item['dates'] == plan['splits'][stage] and item['fit_rows'] == spec['fit_rows'] and item['query_rows'] == spec['query_rows'], 'Report stage contract differs')
        if stage in results:
            path = OUT / 'evaluation' / stage / 'results.json'; model = OUT / 'models' / stage / 'metadata.json'
            need(item['status'] == 'evaluated' and item['result_path'] == rel(path) and item['result_sha256'] == sha(path) and item['metadata_path'] == rel(model) and item['metadata_sha256'] == sha(model), 'Report completed artifact differs')
            need(item['metrics'] == results[stage]['metrics'] and item['decision'] == results[stage]['decision'], 'Report differs from saved result')
        else:
            need(item['status'] == 'not_required' and item['metrics'] is None and item['decision'] is None and item['result_sha256'] is None and item['metadata_sha256'] is None, 'Unexecuted stage is misrepresented')
    mapping(report['receipt_verified_inputs'], 'report receipt-verified input')
    mapping(report['document_sha256'], 'report referenced document')
    add('src/round7_report.py', report['report_script_sha256'], 'report generator')


def collect_files(expected, add):
    files, omitted = set(), []
    for folder, dirs, names in os.walk(OUT, followlinks=False):
        parent = Path(folder); keep = []
        for name in dirs:
            path = parent / name
            if name in OMIT_DIRS or name.startswith(('.', 'live_status')) or path.is_symlink(): omitted.append(rel(path))
            else: keep.append(name)
        dirs[:] = keep
        for name in names:
            path = parent / name
            if path.is_symlink() or name.startswith(('.', 'live_status')) or path.suffix.lower() in {'.pyc', '.tmp', '.temp', '.log', '.lock', '.part'}: omitted.append(rel(path))
            else:
                need(path.suffix.lower() in SUFFIXES, 'Unclassified R7 artifact: ' + rel(path))
                files.add(safe(ROOT, path))
    files.update(safe(ROOT, name) for name in REQUIRED_REPORTS)
    files.update(p.resolve() for p in (ROOT / 'reports').glob('*round7*') if p.is_file())
    roots = set((ROOT / 'src').glob('round7*.py')) | {Path(__file__).resolve()}
    for path in source_closure(roots):
        if path.name.startswith('round7') or path.resolve() == Path(__file__).resolve(): files.add(path)
        else: add(path, reason='older imported source; exact-SHA external')
    for path in ROOT.glob('requirements*.txt'): add(path, reason='existing runtime requirements; external')
    for path in files:
        name = rel(path)
        need(name.startswith(('outputs/round7/', 'reports/')) or name == 'README_round7.md' or name.startswith('src/'), 'Unexpected copy scope: ' + name)
        if name.startswith('src/'): need(path.name.startswith('round7') or path.name == 'archive_round7.py', 'Older source must remain external')
        if name.startswith('reports/'): need('round7' in path.name, 'Older report must remain external')
        need(path.suffix.lower() not in {'.xlsx', '.pbf'}, 'Raw/large map copying prohibited')
    return sorted(files), sorted(omitted)


def file_record(path):
    path = safe(ROOT, path)
    return {'original_path':rel(path), 'original_absolute_path':str(path), 'sha256':sha(path), 'bytes':path.stat().st_size}


def build_archive_plan():
    expected, reasons, add, mapping, locked = registry()
    plan, frozen, accepted, results, metadata, completed = validate_contracts(add, mapping, locked)
    prior = verify_prior(plan, add, locked)
    validate_reports(plan, frozen, accepted, results, metadata, add, mapping)
    files, omitted = collect_files(expected, add)
    originals = {rel(path):file_record(path) for path in files}
    external = []
    for name, digest in sorted(expected.items()):
        if name in originals:
            need(originals[name]['sha256'] == digest, 'Original changed while collecting: ' + name)
        else:
            item = file_record(name); need(item['sha256'] == digest, 'Dependency changed while collecting: ' + name)
            item['recorded_sha256'] = digest; item['reasons'] = sorted(reasons[name]); external.append(item)
    return {'originals':originals, 'external_dependencies':external, 'omitted_paths':omitted,
        'plan_sha256':sha(OUT / 'candidate_plan.json'), 'frozen_candidate_sha256':sha(OUT / 'frozen_candidate.json'), 'accepted_candidate_sha256':sha(OUT / 'accepted_candidate.json'),
        'completed_stages':completed, 'accepted':accepted['accepted'], 'selected_candidate':frozen['selected_candidate'], 'deployment_decision':accepted['deployment_decision'],
        'saved_decisions':{s:v['decision'] for s,v in results.items()}, 'prior_archives_verified':prior,
        'new_release':False, 'new_submission_created':False, 'recommended_submission':'outputs/submission_v3.csv',
        'checks':{'all_receipts_verified':True, 'parameters_and_formulas_verified':True, 'completed_stage_authorization_verified':True, 'input_output_model_sha_verified':True, 'model_indices_verified':True, 'native_model_tree_inventory_verified':True, 'report_saved_results_equal':True, 'prior_releases_and_registry_unchanged':True, 'new_scores_computed':False},
        'dependency_policy':'Only current R7 sources/artifacts/documents copied; all prior source/plan/stage/final/maps/dependencies remain exact-SHA external'}


def verify_snapshot(folder):
    folder = safe(ROOT, folder); path = folder / MANIFEST
    need(path.is_file() and (folder / RECEIPT).is_file(), 'Snapshot or receipt missing')
    need((folder / RECEIPT).read_text(encoding='ascii').strip() == sha(path), 'Snapshot receipt differs')
    manifest = read(path); plan = manifest['original_plan']
    need(manifest['experiment_round'] == 7 and manifest['snapshot_kind'] == 'round7_experiments' and manifest['is_release'] is False, 'Wrong snapshot identity')
    need(manifest['accepted'] == plan['accepted'] and manifest['completed_stages'] == plan['completed_stages'], 'Snapshot outcome differs')
    need(set(manifest['copied_files']) == set(plan['originals']), 'Copy inventory differs')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    need(actual == set(plan['originals']) | {MANIFEST, RECEIPT, NOTICE_FILE}, 'Snapshot actual inventory differs')
    for name, item in manifest['copied_files'].items():
        original = plan['originals'][name]
        need(item['original_path'] == name and item['backup_path'] == 'runs/round7_experiments/' + name, 'Snapshot file path differs')
        need(item['original_sha256'] == item['backup_sha256'] == original['sha256'] and item['original_bytes'] == item['backup_bytes'] == original['bytes'], 'Original/copy proof differs')
        verify_file(folder, name, item['backup_sha256'], item['backup_bytes'])
    need(manifest['external_dependencies'] == plan['external_dependencies'], 'External dependency inventory differs')
    need((folder / NOTICE_FILE).read_bytes() == NOTICE.encode('utf-8'), 'Snapshot dependency notice differs')
    return manifest


def verify_current(plan):
    for name, item in plan['originals'].items(): verify_file(ROOT, name, item['sha256'], item['bytes'])
    for item in plan['external_dependencies']: verify_file(ROOT, item['original_path'], item['sha256'], item['bytes'])
    need(build_archive_plan() == plan, 'Current contracts, sources or artifact inventory changed')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Read-only original/copy/dependency/contract verification')
    args = parser.parse_args(); destination = safe(ROOT, DEST); runs = safe(ROOT, 'runs')
    need(destination.parent == runs and destination.name == 'round7_experiments', 'Unsafe destination')
    if args.verify:
        manifest = verify_snapshot(destination); verify_current(manifest['original_plan'])
        print(json.dumps({'status':'verified', 'snapshot':rel(destination), 'files':len(manifest['copied_files']), 'completed_stages':manifest['completed_stages'], 'manifest_sha256':sha(destination / MANIFEST)})); return
    need(not destination.exists(), 'Snapshot exists; use --verify. Never overwrite')
    plan = build_archive_plan(); runs.mkdir(parents=True, exist_ok=True)
    total_bytes = sum(v['bytes'] for v in plan['originals'].values())
    need(shutil.disk_usage(runs).free > total_bytes + 16 * 1024 * 1024, 'Insufficient disk space')
    staging = safe(runs, '.round7-staging-' + uuid.uuid4().hex)
    need(staging.parent == runs, 'Unsafe staging path'); staging.mkdir()
    try:
        copied = {}
        for name, item in sorted(plan['originals'].items()):
            source = safe(ROOT, name); target = safe(staging, name); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); verify_file(staging, name, item['sha256'], item['bytes'])
            copied[name] = {'original_path':name, 'original_absolute_path':str(source), 'original_sha256':item['sha256'], 'original_bytes':item['bytes'], 'backup_path':'runs/round7_experiments/' + name, 'backup_sha256':sha(target), 'backup_bytes':target.stat().st_size}
        (staging / NOTICE_FILE).write_bytes(NOTICE.encode('utf-8'))
        manifest = {'format_version':1, 'snapshot_kind':'round7_experiments', 'experiment_round':7, 'is_release':False, 'accepted':plan['accepted'], 'completed_stages':plan['completed_stages'], 'selected_candidate':plan['selected_candidate'], 'created_at_utc':datetime.now(timezone.utc).isoformat(), 'copied_files':copied, 'external_dependencies':plan['external_dependencies'], 'original_plan':plan, 'self_contained':False, 'dependency_notice':NOTICE_FILE, 'new_submission_created':False, 'versions_modified':False, 'prior_archives_modified':False, 'uploaded_to_kaggle':False, 'manifest_self_excluded':True, 'manifest_sha256_receipt':RECEIPT}
        (staging / MANIFEST).write_bytes(encoded(manifest))
        (staging / RECEIPT).write_text(sha(staging / MANIFEST) + '\n', encoding='ascii')
        verify_snapshot(staging); verify_current(plan)
        need(not destination.exists() and staging.parent == destination.parent == runs, 'Destination appeared or unsafe publication path')
        need(sha(ROOT / 'runs/versions.json') == plan['prior_archives_verified']['registry_sha256'], 'Registry changed before publication')
        staging.rename(destination)
    except BaseException:
        print('Archive not finalized; staging retained: ' + str(staging), flush=True); raise
    verify_snapshot(destination)
    print(json.dumps({'status':'created', 'snapshot':rel(destination), 'files':len(copied), 'copied_bytes':total_bytes, 'external_dependencies':len(plan['external_dependencies']), 'completed_stages':plan['completed_stages'], 'accepted':plan['accepted'], 'manifest_sha256':sha(destination / MANIFEST), 'versions_modified':False, 'new_submission_created':False}))


if __name__ == '__main__': main()
