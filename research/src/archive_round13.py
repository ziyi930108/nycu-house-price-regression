"""Archive the finalized R13 development rejection; stdlib and byte hashes only.
Explicit --create / --verify / --self-check. No model or array deserialization,
training, scoring, release, registry write, cleanup, or old-source imports.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round13'
DEST = ROOT / 'runs/round13_experiments'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, CANDIDATE = 'v3_control', 'city_alias_geography_blend'
SOURCES = ('src/round13_city_alias_geo.py', 'src/round13_experiment.py',
           'src/round13_verify_backtest.py', 'src/round13_report.py',
           'src/round13_alias_inventory.py', 'src/archive_round13.py')
DOCS = ('reports/round13_plan.md', 'reports/round13_city_alias_feasibility.md',
        'reports/round13_official_name_evidence.md', 'reports/round13_alias_inventory.md',
        'reports/round13_pipeline_review.md', 'reports/round13_work_log.md',
        'reports/round13_report.md', 'reports/round13_results.json', 'reports/round13_results.sha256')
RESEARCH = ROOT / 'outputs/round13_research'
RESEARCH_FILES = ('outputs/round13_research/alias_inventory.json',
                  'outputs/round13_research/alias_inventory.sha256')
ALIAS_COLUMN = 'r13_address_city_canonical'
ALIAS_MAPPING = {'桃園縣': '桃園市'}
MODEL_FILES = {'model.cbm', 'feature_state.json', 'predictions.npy', 'raw_z.npy',
               'fit_indices.npy', 'query_indices.npy', 'metadata.json', 'metadata.sha256'}
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = '''# R13 development-rejection snapshot

This utility supports only a finalized rejection by the fixed development gates.
The archived outcome is not assumed until current authority receipts are checked.
Confirmation and auxiliary price fits/evaluations must not have been performed.
The recommendation remains outputs/submission_v3.csv; this is not a release.

R13 sources, documents, finalized outputs, any preparation attempts/logs and the
separate target-free alias inventory are copied byte for byte. The raw-price
component uses 96 features / 24 categories: unchanged 95 R6 features plus the
fixed city alias. Historical diagnostic bytes are not current source declarations.

Original data, earlier models, all four annual geographic dependency chains and
other declared dependencies remain external, checked by SHA-256 and byte length.
The pure-X inventory source/input maps are included. This is not self-contained.
No array/model is deserialized and no price error or model decision is recomputed.

Current originals of frozen copied files are checked. versions.json is only
creation-time context; it and CURRENT_STATUS.md are not permanent guards. No
registry, submission, older experiment, or later-round artifact is written.
Partial model/evaluation/report/research/archive staging is refused and retained.
'''


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def normalized(name):
    need(isinstance(name, str), 'Path must be a relative string')
    name = name.replace('\\', '/')
    need(name and name == name.strip() and ':' not in name and '\x00' not in name,
         'Unsafe relative path: ' + repr(name))
    need(not PurePosixPath(name).is_absolute()
         and all(part not in ('', '.', '..') for part in name.split('/')),
         'Unsafe relative path: ' + name)
    return name


def safe(name, base=ROOT, exists=True):
    name = normalized(name)
    root = Path(base).resolve()
    path = (root / name).resolve()
    need(path.is_relative_to(root), 'Path escapes declared root: ' + name)
    if exists:
        need(path.is_file(), 'Missing file: ' + str(path))
    return path


def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()


def locked(path):
    path = Path(path)
    receipt = path.with_suffix('.sha256')
    need(path.is_file() and receipt.is_file(), 'Missing JSON/receipt: ' + str(path))
    need(receipt.read_text(encoding='ascii').strip() == sha(path), 'Receipt differs: ' + str(path))
    return read(path)


def copy_scope(name):
    name = normalized(name)
    return (name in SOURCES or (name.startswith('src/round13_') and name.endswith('.py'))
            or name.startswith('outputs/round13/') or name.startswith('outputs/round13_research/')
            or name.startswith('reports/round13_') or name == 'README_round13.md')


class Bindings:
    """Merge only authoritative current maps, with mutable exceptions explicit."""
    def __init__(self):
        self.values, self.case_names, self.mutable_declarations = {}, {}, {}

    def declare(self, name, digest):
        name = normalized(name)
        need(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid SHA: ' + name)
        previous = self.case_names.get(name.casefold())
        need(previous is None or previous == name, 'Case-ambiguous dependency: ' + name)
        self.case_names[name.casefold()] = name
        if name.casefold() in {p.casefold() for p in MUTABLE}:
            need(name in MUTABLE, 'Case-ambiguous mutable path')
            self.mutable_declarations.setdefault(name, set()).add(digest)
            return False
        need(name not in self.values or self.values[name]['sha256'] == digest,
             'Conflicting authority SHA: ' + name)
        return True

    def add(self, name, expected=None):
        name = normalized(name)
        if expected is not None and not self.declare(name, expected):
            return None  # Historical context, deliberately not a live guard.
        if name in self.values:
            return safe(name)
        path = safe(name)
        digest = sha(path)
        if not self.declare(name, digest):
            return None
        need(expected is None or expected == digest, 'Changed dependency: ' + name)
        self.values[name] = {'sha256': digest, 'bytes': path.stat().st_size}
        return path

    def mapping(self, values):
        need(isinstance(values, dict), 'Expected path-to-SHA map')
        for name, digest in values.items():
            self.add(name, digest)

    def authority(self, name):
        path = self.add(name)
        need(path is not None, 'Mutable file cannot be authority')
        value = locked(path)
        self.add(rel(path.with_suffix('.sha256')))
        return value


def inventory():
    need(OUT.is_dir() and RESEARCH.is_dir(), 'Missing R13 formal/research outputs')
    for area in ('models', 'evaluation'):
        folder = OUT / area
        need(folder.is_dir() and {p.name for p in folder.iterdir()} == {'development'},
             'Unexpected/incomplete/unauthorized stage in ' + area)
    for name in ('final', 'release'):
        need(not (OUT / name).exists(), 'R13 rejection cannot contain ' + name)
    need(not list((ROOT / 'reports').glob('.round13-report-*')),
         'Partial report staging requires inspection; keep it')
    need(not list(RESEARCH.glob('.alias_inventory-*')),
         'Partial research publication requires inspection; keep it')
    model = OUT / 'models/development'
    evaluation = OUT / 'evaluation/development'
    need({p.name for p in model.iterdir()} == MODEL_FILES
         and all(p.is_file() for p in model.iterdir()), 'Incomplete model inventory')
    need({p.name for p in evaluation.iterdir()} == {CONTROL + '.npy', CANDIDATE + '.npy',
                                                 'results.json', 'results.sha256'}
         and all(p.is_file() for p in evaluation.iterdir()), 'Incomplete evaluation inventory')
    paths = set(OUT.rglob('*')) | set(RESEARCH.rglob('*'))
    paths.update((ROOT / 'src').glob('round13_*.py'))
    for path in list((ROOT / 'reports').glob('round13_*')):
        paths.add(path)
        if path.is_dir():
            paths.update(path.rglob('*'))
    paths.update(ROOT / name for name in SOURCES)
    if (ROOT / 'README_round13.md').exists():
        paths.add(ROOT / 'README_round13.md')
    files = set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink is not an archive artifact: ' + str(path))
        if path.is_file():
            name = rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R13 copy scope')
            if name.startswith('reports/'):
                need(not name.lower().endswith(('.tmp', '.part', '.lock')), 'Partial report file')
            need(not (path.name.lower().startswith('submission') and path.suffix.lower() == '.csv'),
                 'A backtest archive must not contain a new submission')
            files.add(name)
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= files, 'Required R13 source/document missing')
    return files


def rejection_contract(plan, frozen, accepted, result):
    need(plan['round'] == frozen['round'] == accepted['round'] == result['round'] == 13,
         'Wrong experiment round')
    need(plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong candidate/control')
    need(result['stage'] == 'development' and result['evaluated_candidates'] == [CONTROL, CANDIDATE],
         'Only development may have been evaluated')
    need(result['decision']['passed'] is False and result['decision']['selected_candidate'] == CONTROL,
         'This bounded archiver only accepts a development rejection')
    need(frozen['passed'] is False and frozen['later_stages_allowed'] is False
         and frozen['selected_candidate'] == CONTROL, 'Development freeze differs')
    need(accepted['accepted'] is False and accepted['selected_candidate'] == CONTROL
         and accepted['stopping_reason'] == 'development_price_gates_failed'
         and accepted['deployment_decision'] == 'keep_v3', 'Unexpected final decision')
    need(accepted['stage_status'] == {s: 'price_evaluated' if s == 'development'
         else 'not_executed_after_development_rejection' for s in STAGES}
         and accepted['unevaluated_stages'] == list(STAGES[1:]), 'Wrong stage status inventory')
    need(set(accepted['stage_result_sha256']) == {'development'}, 'Unexpected later result')
    need(accepted['new_submission_created'] is False
         and accepted['currently_recommended_submission'] == 'outputs/submission_v3.csv'
         and accepted['no_reselection_or_parameter_change'] is True,
         'Unexpected release or reselection')
    for value in (plan, frozen, accepted, result):
        need(value['historically_untouched'] is False, 'Historical reuse must remain explicit')
    need(plan['feature_count'] == 96 and plan['categorical_count'] == 24
         and plan['parameters']['iterations'] == 2999 and plan['parameters']['random_seed'] == 42
         and plan['sample_weight'] == 'uniform' and plan['target_mode'] == 'raw',
         'Fixed model contract differs')
    need(plan['target_formula'] == 'y_TWD / 1e6'
         and plan['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong target/inverse')
    need(plan['component'] == 'geography_city_alias_raw_catboost', 'Wrong raw geographic component')
    rules = plan['feature_rules']
    need(rules['alias_mapping'] == ALIAS_MAPPING and rules['new_categorical_column'] == ALIAS_COLUMN
         and rules['original_features_unchanged'] == 87 and rules['existing_geo_features_unchanged'] == 8
         and rules['existing_95_values_dtypes_order_and_index_unchanged'] is True
         and rules['reference_scoring_grouping_or_dedup_keys_changed'] is False,
         'Wrong additive-only alias rules')
    need(plan['alias_uses_query_statistics_or_y'] is False
         and plan['original_features_and_scoring_keys_preserved'] is True,
         'Alias or original reference/grouping contract changed')
    need(result['fit_rows'] == plan['split_contract']['development']['fit_rows']
         and result['query_rows'] == plan['split_contract']['development']['query_rows'], 'Rows differ')
    for field in ('formulas', 'thresholds'):
        need(result[field] == frozen[field] == plan[field], 'Frozen protocol differs: ' + field)
    need(frozen['parameters'] == plan['parameters'] and result['target_group_weights']
         == frozen['target_group_weights'] == plan['target_group_weights'], 'Frozen parameters/weights differ')


def check_report(report, plan, accepted, result, model, proof, proof_sha):
    need(report['round'] == 13 and report['accepted'] is False
         and report['stopping_reason'] == accepted['stopping_reason']
         and report['deployment_decision'] == 'keep_v3', 'Report decision differs')
    for key in ('new_submission_created', 'historically_untouched', 'public_score_used', 'private_score_known'):
        need(report[key] is False, 'Report policy differs: ' + key)
    need(report['report_only_reads_saved_JSON_and_hashes_binary_inputs'] is True, 'Report read-scope differs')
    need([item['stage'] for item in report['stages']] == list(STAGES), 'Report stage inventory differs')
    fields = ('iterations', 'feature_count', 'categorical_feature_count', 'sample_weight',
              'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds')
    for item in report['stages']:
        stage = item['stage']; spec = plan['split_contract'][stage]
        need(item['status'] == accepted['stage_status'][stage] and item['dates'] == plan['splits'][stage]
             and item['fit_rows'] == spec['fit_rows'] and item['query_rows'] == spec['query_rows'],
             'Report stage identity/count differs')
        if stage == 'development':
            need(item['metrics'] == result['metrics'] and item['decision'] == result['decision'],
                 'Report changed saved metrics/decision')
            need(item['model'] == {key: model[key] for key in fields}, 'Report model summary differs')
            need(item['independent_review'] == {
                'proof_path': 'outputs/round13/audit/development_replay.json', 'proof_sha256': proof_sha,
                'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal']},
                'Report replay proof differs')
        else:
            need(all(item[key] is None for key in ('metrics', 'decision', 'model', 'independent_review')),
                 'Report invents unexecuted scores or models')



def check_research(research, plan):
    need(research['round'] == 13 and research['input_and_source_hashes_unchanged_after_read'] is True,
         'Unverified pure-X inventory')
    bounds = research['read_boundaries']
    need(bounds['pure_X_cache_only'] is True and bounds['development_query_X_read'] is True
         and bounds['historical_target_containing_dedup_mapping_reused'] is True,
         'Pure-X inventory alignment/exposure differs')
    for key in ('raw_workbooks_read', 'target_cache_read', 'labels_predictions_or_scores_read',
                'test_X_read', 'development_query_is_untouched_blind',
                'model_fit_or_scoring_performed', 'alias_mapping_inferred_or_applied'):
        need(bounds[key] is False, 'Unexpected inventory read/use boundary: ' + key)
    need(research['scope']['exact_parent_city_values'] == ['桃園縣', '桃園市']
         and set(research['phases']) == {'development_fit', 'development_query'},
         'Unexpected inventory scope')
    spec = plan['split_contract']['development']
    alignment = research['alignment']
    for key in ('fit_rows', 'query_rows', 'fit_indices_content_sha256', 'query_indices_content_sha256'):
        need(alignment[key] == spec[key], 'Inventory canonical split differs: ' + key)
    need(alignment['indices_dtype_unique_increasing_range_order_and_dates_verified'] is True
         and alignment['fit_query_overlap_rows'] == 0, 'Inventory row invariants differ')
    need(research['report']['path'] == 'reports/round13_alias_inventory.md', 'Wrong inventory report path')
    required_sources = {'src/round13_alias_inventory.py', 'src/house_features.py', 'src/round6_prepare_geo.py'}
    required_inputs = {'outputs/round6/candidate_plan.json', 'outputs/round6/candidate_plan.sha256',
                       'outputs/round6/geography/development/manifest.json',
                       'outputs/round6/geography/development/manifest.sha256',
                       'data_cache/train_features.pkl', 'reports/feature_schema.json',
                       'outputs/round6/geography/development/train_source_cache_indices.npy',
                       'outputs/round6/models/development/geography_catboost/fit_indices.npy',
                       'outputs/round6/models/development/geography_catboost/query_indices.npy',
                       'reports/round13_city_alias_feasibility.md', 'reports/round13_official_name_evidence.md'}
    need(set(research['source_sha256']) == required_sources
         and set(research['input_sha256']) == required_inputs, 'Inventory source/input closure differs')
    need(research['input_sha256']['outputs/round6/candidate_plan.json'] == plan['old_plan_sha256'],
         'Inventory references another historical plan')


def collect():
    """Read finalized authorities and hashes. Never inspect npy/model contents."""
    files = inventory()
    b = Bindings()
    authority_paths = {
        'plan': 'outputs/round13/candidate_plan.json',
        'frozen': 'outputs/round13/frozen_candidate.json',
        'accepted': 'outputs/round13/accepted_candidate.json',
        'result': 'outputs/round13/evaluation/development/results.json',
        'model': 'outputs/round13/models/development/metadata.json',
        'proof': 'outputs/round13/audit/development_replay.json',
        'report': 'reports/round13_results.json',
    }
    values = {key: b.authority(name) for key, name in authority_paths.items()}
    plan, frozen, accepted, result, model, proof, report = (values[k] for k in authority_paths)
    digests = {key: b.values[name]['sha256'] for key, name in authority_paths.items()}
    rejection_contract(plan, frozen, accepted, result)
    for value in (frozen, accepted, result, model, proof):
        need(value['plan_sha256'] == digests['plan'], 'Authority refers to another R13 plan')
    need(accepted['frozen_candidate_sha256'] == digests['frozen']
         and frozen['result_sha256'] == proof['result_sha256'] == digests['result']
         and accepted['stage_result_sha256'] == {'development': digests['result']},
         'Finalized result lineage differs')
    for value in (frozen, accepted, result, model):
        need(value['source_sha256'] == plan['source_sha256'], 'Authority source closure differs')
    for field in ('source_sha256', 'dependency_sha256', 'final_dependency_sha256'):
        b.mapping(plan[field])
    need(set(plan['stage_dependency_sha256']) == set(STAGES), 'Incomplete annual dependency maps')
    for mapping in plan['stage_dependency_sha256'].values():
        b.mapping(mapping)
    b.mapping(frozen['development_artifact_sha256'])
    for field in ('input_artifact_sha256', 'output_artifact_sha256'):
        b.mapping(result[field])
    b.mapping(proof['source_input_sha256'])
    b.mapping(report['source_input_sha256'])
    b.mapping(proof['protected_artifacts_unchanged'])
    b.mapping(model['component_source_sha256'])
    b.mapping(model['mapping_evidence_sha256'])
    research = b.authority(RESEARCH_FILES[0])
    check_research(research, plan)
    for name in (*RESEARCH_FILES, 'src/round13_alias_inventory.py', 'reports/round13_alias_inventory.md'):
        b.add(name)
        need(plan['dependency_sha256'].get(name) == b.values[name]['sha256'],
             'Pure-X research artifact not bound by candidate plan: ' + name)
    b.mapping(research['source_sha256']); b.mapping(research['input_sha256'])
    b.add(research['report']['path'], research['report']['sha256'])
    # All four geography chains are already plan-bound, including unused later
    # maps. Assert the expansion is complete; never silently drop an edge.
    for stage in (*STAGES, 'final'):
        mapping = plan['final_dependency_sha256'] if stage == 'final' else plan['stage_dependency_sha256'][stage]
        name = 'outputs/round6/geography/' + stage + '/manifest.json'
        old = b.authority(name)
        need(mapping.get(name) == b.values[name]['sha256'], 'Geography authority missing from plan')
        for field in ('source_sha256', 'index_files_sha256'):
            need(all(mapping.get(k) == v for k, v in old[field].items()), 'Geography source/index closure omitted')
            b.mapping(old[field])
        for item in old['features'].values():
            need(mapping.get(item['path']) == item['sha256'], 'Geography feature missing from authority')
            b.add(item['path'], item['sha256'])
    mf = OUT / 'models/development'
    need(set(model['artifact_sha256']) == MODEL_FILES - {'metadata.json', 'metadata.sha256'},
         'Model metadata inventory differs')
    for name, digest in model['artifact_sha256'].items():
        need(Path(name).name == name, 'Unsafe model artifact name')
        b.add(rel(mf / name), digest)
    expected_outputs = {'outputs/round13/evaluation/development/' + name + '.npy'
                        for name in (CONTROL, CANDIDATE)}
    need(set(result['output_artifact_sha256']) == expected_outputs, 'Unexpected prediction inventory')
    need(model['round'] == 13 and model['stage'] == 'development'
         and model['component'] == plan['component'] and model['parameters'] == plan['parameters'],
         'Model configuration differs')
    need(model['iterations'] == model['actual_iterations'] == 2999
         and model['feature_count'] == 96 and model['categorical_feature_count'] == 24
         and model['sample_weight'] == 'uniform' and model['target_mode'] == plan['target_mode']
         and model['target_formula'] == plan['target_formula']
         and model['inverse_formula'] == plan['inverse_formula']
         and model['feature_rules'] == plan['feature_rules'], 'Model scale/schema differs')
    spec = plan['split_contract']['development']
    need(model['outer_fit_rows'] == model['fit_rows'] == proof['fit_rows'] == spec['fit_rows']
         and model['query_rows'] == proof['query_rows'] == spec['query_rows'], 'Model/proof counts differ')
    for role in ('fit', 'query'):
        need(model[role + '_indices_content_sha256'] == spec[role + '_indices_content_sha256'],
             'Saved index identity differs from frozen plan')
    need(model['prediction_units'] == 'TWD' and model['reload_all_query_rows_verified'] is True
         and model['reload_exact_equality'] is True and model['reload_query_rows'] == spec['query_rows'],
         'Model reload evidence differs')
    for key in ('query_labels_used', 'fit_uses_eval_set', 'early_stopping_used',
                'calibration_applied', 'extra_price_cap_applied', 'reference_or_trend_applied_here', 'reference_features_changed'):
        need(model[key] is False, 'Unexpected model behavior: ' + key)
    need(proof['round'] == 13 and proof['stage'] == 'development' and proof['status'] == 'passed'
         and proof['candidate_accepted_in_this_stage'] is False and proof['saved_raw_z_bitwise_equal'] is True,
         'Independent proof does not certify this rejection')
    need(proof['raw_z_units'] == 'million_TWD' and proof['alias_column'] == ALIAS_COLUMN
         and proof['alias_mapping'] == ALIAS_MAPPING
         and proof['model_view_reimplemented_independently'] is True
         and proof['original_95_values_dtypes_order_index_preserved'] is True
         and proof['original_87_reference_and_scoring_keys_unchanged'] is True,
         'Independent proof does not match additive raw alias contract')
    need(proof['independent_metrics'] == result['metrics']
         and proof['independent_decision'] == result['decision']
         and proof['raw_z_diagnostics'] == model['raw_z_diagnostics'], 'Proof/report payload mismatch')
    for key in ('fit_performed', 'new_window_scored', 'public_score_used', 'historically_untouched'):
        need(proof[key] is False, 'Unexpected independent review behavior: ' + key)
    b.add('src/round13_verify_backtest.py', proof['source_sha256'])
    check_report(report, plan, accepted, result, model, proof, digests['proof'])
    # These three source maps are authoritative; verify that the independent
    # replay actually bound the artifacts it says it replayed, without loading them.
    required_proof = {authority_paths[k] for k in ('plan', 'frozen', 'result', 'model')}
    required_proof.update('outputs/round13/models/development/' + name
                          for name in MODEL_FILES)
    required_proof.update(expected_outputs)
    need(required_proof <= set(proof['source_input_sha256']), 'Replay input closure is incomplete')
    required_report = set(required_proof) | {authority_paths['accepted'], authority_paths['proof'],
                                           'src/round13_report.py'}
    need(required_report <= set(report['source_input_sha256']), 'Report input closure is incomplete')
    # Attempts are copied, not interpreted as current source bindings.
    for name in sorted(files):
        b.add(name)
        if name.endswith('.sha256'):
            path = safe(name)
            target = path.with_suffix('.json')
            need(target.is_file(), 'Historical/current receipt has no JSON: ' + name)
            locked(target)  # Byte receipt only; do not traverse historical source maps.
    need(inventory() == files, 'R13 inventory changed during collection')
    need(all(not copy_scope(name) or name in files for name in b.values), 'Owned dependency omitted from copy inventory')
    copied = {name: b.values[name].copy() for name in sorted(files)}
    external = {name: item.copy() for name, item in sorted(b.values.items()) if name not in copied}
    need(not MUTABLE.intersection(set(copied) | set(external)), 'Mutable entry point became permanent')
    registry = safe('runs/versions.json')
    context = {'path': 'runs/versions.json', 'sha256': sha(registry), 'bytes': registry.stat().st_size,
               'plan_time_sha256': plan['registry_sha256_at_plan'],
               'authority_observed_sha256': sorted(b.mutable_declarations.get('runs/versions.json', set())),
               'verification_policy': 'creation-time observation only; never a future external guard'}
    state = {'round': 13, 'is_release': False, 'accepted_backtest': False,
             'candidate': CANDIDATE, 'selected_candidate': CONTROL,
             'stopping_reason': accepted['stopping_reason'], 'stage_status': accepted['stage_status'],
             'recommended_version': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
             'new_submission_created': False, 'registry_modified': False,
             'plan_sha256': digests['plan'], 'frozen_candidate_sha256': digests['frozen'],
             'accepted_candidate_sha256': digests['accepted'], 'development_result_sha256': digests['result'],
             'independent_replay_sha256': digests['proof'], 'report_json_sha256': digests['report'],
             'research_inventory_sha256': b.values[RESEARCH_FILES[0]]['sha256'],
             'research_source_and_input_maps_closed': True,
             'source_path': 'src/archive_round13.py', 'source_sha256': sha(__file__),
             'property_labels_or_prediction_values_read_by_archiver': False,
             'array_or_model_deserialization_performed': False, 'models_refitted_or_MAE_recomputed': False,
             'index_identity_uses_saved_contract_and_replay_not_array_deserialization': True,
             'historically_untouched': False, 'historical_attempts_kept_without_current_source_enforcement': True,
             'registry_context': context, 'current_status_is_not_bound': True}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round13_development_rejection_experiments', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 13 and state['is_release'] is False and state['accepted_backtest'] is False
         and state['selected_candidate'] == CONTROL and state['stopping_reason'] == 'development_price_gates_failed'
         and state['recommended_version'] == 'v3' and state['new_submission_created'] is False,
         'Snapshot does not describe the finalized development rejection')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copied and external paths overlap')
    names = set(copied) | set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names} & {n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= set(copied) and all(copy_scope(n) for n in copied),
         'Wrong R13 copy inventory')
    need(all(not copy_scope(n) for n in external), 'Owned R13 artifact left external')
    need(manifest['copied_file_count'] == len(copied)
         and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(item['bytes'] for item in copied.values()),
         'Manifest count/size mismatch')
    for name, item in {**copied, **external}.items():
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid file identity: ' + name)
    for key, name in [('plan_sha256', 'outputs/round13/candidate_plan.json'),
                      ('frozen_candidate_sha256', 'outputs/round13/frozen_candidate.json'),
                      ('accepted_candidate_sha256', 'outputs/round13/accepted_candidate.json'),
                      ('development_result_sha256', 'outputs/round13/evaluation/development/results.json'),
                      ('independent_replay_sha256', 'outputs/round13/audit/development_replay.json'),
                      ('report_json_sha256', 'reports/round13_results.json'),
                      ('research_inventory_sha256', RESEARCH_FILES[0]),
                      ('source_sha256', 'src/archive_round13.py')]:
        need(state[key] == copied[name]['sha256'], 'Manifest authority identity mismatch: ' + key)
    return copied, external


def verify_at(folder, originals=True):
    """Verify a snapshot using only JSON and raw bytes, in a fresh process."""
    folder = Path(folder).resolve()
    need(folder.is_relative_to((ROOT / 'runs').resolve()), 'Snapshot outside workspace runs')
    manifest = locked(folder / 'snapshot_manifest.json')
    copied, external = manifest_structure(manifest)
    expected = {'snapshot_manifest.json', 'snapshot_manifest.sha256', 'SNAPSHOT_README.md'}
    for name, item in copied.items():
        need(item['backup_path'] == rel(DEST / 'files' / name), 'Unexpected final backup path')
        backup = safe('files/' + normalized(name), folder)
        need(backup.stat().st_size == item['bytes'] and sha(backup) == item['sha256'], 'Snapshot copy changed: ' + name)
        if originals:
            original = safe(name)
            need(original.stat().st_size == item['bytes'] and sha(original) == item['sha256'], 'Frozen original changed: ' + name)
        expected.add('files/' + name)
    for name, item in external.items():
        path = safe(name)
        need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'External dependency changed: ' + name)
    notice = folder / 'SNAPSHOT_README.md'
    need(notice.read_bytes() == NOTICE.encode('utf-8') and sha(notice) == manifest['notice_sha256'], 'Snapshot notice differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == expected,
         'Snapshot contains missing/extra files')
    if originals:
        need(inventory() == set(copied), 'Original R13 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}


def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round13_experiments_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round13_experiments_staging_' + uuid.uuid4().hex)
    need(staging.resolve().is_relative_to((ROOT / 'runs').resolve()) and not staging.exists(), 'Unsafe staging')
    staging.mkdir(parents=True)
    records = {}
    for name, item in copied.items():
        source, target = safe(name), safe('files/' + name, staging, exists=False)
        need(not target.exists(), 'Copy destination exists')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        need(source.stat().st_size == target.stat().st_size == item['bytes']
             and sha(source) == sha(target) == item['sha256'], 'Source/copy changed: ' + name)
        records[name] = {**item, 'original_sha256': item['sha256'], 'backup_path': rel(DEST / 'files' / name)}
    notice = staging / 'SNAPSHOT_README.md'
    notice.write_bytes(NOTICE.encode('utf-8'))
    manifest = {'kind': 'round13_development_rejection_experiments',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'state': state,
                'copied_files': records, 'external_dependencies': external,
                'copied_file_count': len(records), 'external_dependency_count': len(external),
                'copied_bytes': sum(item['bytes'] for item in records.values()),
                'notice_sha256': sha(notice), 'not_self_contained': True,
                'failed_staging_preserved': True, 'registry_and_mutable_status_not_modified_or_bound': True}
    path = staging / 'snapshot_manifest.json'
    with path.open('xb') as handle:
        handle.write(encoded(manifest))
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(path) + '\n')
    # Verify all bytes/authorities and exact inventory before the final rename.
    verify_at(staging)
    need(not DEST.exists(), 'Concurrent destination appeared; staging preserved')
    staging.rename(DEST)
    print(json.dumps({'status': 'created', 'snapshot_sha256': sha(DEST / 'snapshot_manifest.json'),
                      'copied_files': len(records), 'external_dependencies': len(external),
                      'copied_bytes': manifest['copied_bytes'], 'accepted_backtest': False,
                      'recommended_version': 'v3'}, ensure_ascii=False), flush=True)


def self_check():
    """Bounded synthetic contracts only; no real authority/result/model reads."""
    checks = []
    def check(name, value):
        need(bool(value), 'Synthetic check failed: ' + name); checks.append(name)
    def reject(name, operation):
        try:
            operation()
        except (RuntimeError, KeyError, TypeError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    for name in ('../outside', 'C:/outside', '/outside', 'a/../b', 'a//b', './file', 'file:stream'):
        reject('unsafe_path_' + name, lambda name=name: normalized(name))
    check('windows_relative_separator_normalized', normalized('src\\round13_city_alias_geo.py') == 'src/round13_city_alias_geo.py')
    check('copy_only_R13_owned_artifacts', all(copy_scope(n) for n in SOURCES)
          and copy_scope('outputs/round13/audit/preflight_attempt.json')
          and copy_scope('outputs/round13_research/alias_inventory.json')
          and not any(copy_scope(n) for n in ('src/round12_log_geo.py', 'outputs/round12/audit/a.json',
                                             'reports/round14_plan.md', 'data_external/geography/a.pbf')))
    b = Bindings(); b.declare('src/example.py', 'a'*64)
    b.values['src/example.py'] = {'sha256': 'a'*64, 'bytes': 1}
    check('same_authority_SHA_merges', b.declare('src/example.py', 'a'*64))
    reject('conflicting_authority_rejected', lambda: b.declare('src/example.py', 'b'*64))
    reject('case_alias_rejected', lambda: b.declare('SRC/example.py', 'a'*64))
    reject('invalid_digest_rejected', lambda: b.declare('src/a.py', 'not-a-hash'))
    b.declare('runs/versions.json', 'a'*64); b.declare('runs/versions.json', 'b'*64)
    b.declare('CURRENT_STATUS.md', 'c'*64)
    check('mutable_context_never_enters_dependencies', not MUTABLE.intersection(b.values)
          and len(b.mutable_declarations['runs/versions.json']) == 2)
    plan = {'round': 13, 'candidate': CANDIDATE, 'control': CONTROL, 'historically_untouched': False,
            'feature_count': 96, 'categorical_count': 24, 'parameters': {'iterations': 2999, 'random_seed': 42},
            'component': 'geography_city_alias_raw_catboost',
            'alias_uses_query_statistics_or_y': False, 'original_features_and_scoring_keys_preserved': True,
            'feature_rules': {'alias_mapping': dict(ALIAS_MAPPING), 'new_categorical_column': ALIAS_COLUMN,
                              'original_features_unchanged': 87, 'existing_geo_features_unchanged': 8,
                              'existing_95_values_dtypes_order_and_index_unchanged': True,
                              'reference_scoring_grouping_or_dedup_keys_changed': False},
            'sample_weight': 'uniform', 'target_mode': 'raw', 'target_formula': 'y_TWD / 1e6',
            'inverse_formula': 'max(z * 1e6, 0)', 'formulas': {'fixed': True}, 'thresholds': {'fixed': True},
            'target_group_weights': {'seen': 1.0}, 'splits': {s: ['start', 'end'] for s in STAGES},
            'split_contract': {s: {'fit_rows': 5, 'query_rows': 3} for s in STAGES}}
    frozen = {'round': 13, 'candidate': CANDIDATE, 'passed': False, 'later_stages_allowed': False,
              'selected_candidate': CONTROL, 'historically_untouched': False,
              **{k: copy.deepcopy(plan[k]) for k in ('formulas', 'thresholds', 'parameters', 'target_group_weights')}}
    accepted = {'round': 13, 'candidate': CANDIDATE, 'accepted': False, 'selected_candidate': CONTROL,
                'stopping_reason': 'development_price_gates_failed', 'deployment_decision': 'keep_v3',
                'stage_status': {s: 'price_evaluated' if s == 'development' else
                                 'not_executed_after_development_rejection' for s in STAGES},
                'unevaluated_stages': list(STAGES[1:]), 'stage_result_sha256': {'development': 'a'*64},
                'new_submission_created': False, 'currently_recommended_submission': 'outputs/submission_v3.csv',
                'no_reselection_or_parameter_change': True, 'historically_untouched': False}
    result = {'round': 13, 'stage': 'development', 'evaluated_candidates': [CONTROL, CANDIDATE],
              'decision': {'passed': False, 'selected_candidate': CONTROL}, 'fit_rows': 5, 'query_rows': 3,
              'historically_untouched': False, 'metrics': {'synthetic_saved_value': 1.0},
              **{k: copy.deepcopy(plan[k]) for k in ('formulas', 'thresholds', 'target_group_weights')}}
    rejection_contract(plan, frozen, accepted, result); checks.append('finalized_dev_rejection_supported')
    for name, target, key, value in (
            ('acceptance_cannot_publish', 'accepted', 'accepted', True),
            ('later_stage_status_rejected', 'accepted', 'stage_status', {s: 'price_evaluated' for s in STAGES}),
            ('new_submission_rejected', 'accepted', 'new_submission_created', True),
            ('development_pass_not_supported', 'frozen', 'passed', True),
            ('wrong_scale_rejected', 'plan', 'target_formula', 'log(y)'),
            ('different_training_weights_rejected', 'plan', 'sample_weight', 'importance'),
            ('old_95_feature_model_rejected', 'plan', 'feature_count', 95),
            ('old_23_category_model_rejected', 'plan', 'categorical_count', 23),
            ('log_target_mode_rejected', 'plan', 'target_mode', 'log1p_raw'),
            ('query_statistics_alias_rejected', 'plan', 'alias_uses_query_statistics_or_y', True),
            ('changed_reference_keys_rejected', 'plan', 'original_features_and_scoring_keys_preserved', False),
            ('invented_query_count_rejected', 'result', 'query_rows', 4)):
        fixtures = {k: copy.deepcopy(v) for k,v in [('plan',plan),('frozen',frozen),('accepted',accepted),('result',result)]}
        fixtures[target][key] = value
        reject(name, lambda fixtures=fixtures: rejection_contract(**fixtures))
    model = {k: v for k,v in [('iterations',2999),('feature_count',96),('categorical_feature_count',24),
                              ('sample_weight','uniform'),('target_formula',plan['target_formula']),
                              ('inverse_formula',plan['inverse_formula']),('raw_z_diagnostics',{'synthetic':True}),
                              ('runner_elapsed_seconds',1.0)]}
    proof = {'replays': {'synthetic': True}, 'saved_raw_z_bitwise_equal': True}
    report = {'round':13,'accepted':False,'stopping_reason':accepted['stopping_reason'],'deployment_decision':'keep_v3',
              'new_submission_created':False,'historically_untouched':False,'public_score_used':False,
              'private_score_known':False,'report_only_reads_saved_JSON_and_hashes_binary_inputs':True,'stages':[]}
    for stage in STAGES:
        row = {'stage':stage,'status':accepted['stage_status'][stage],'dates':plan['splits'][stage],
               'fit_rows':5,'query_rows':3,'metrics':None,'decision':None,'model':None,'independent_review':None}
        if stage == 'development':
            row.update(metrics=result['metrics'],decision=result['decision'],model=model,
                       independent_review={'proof_path':'outputs/round13/audit/development_replay.json',
                         'proof_sha256':'a'*64,'replays':proof['replays'],'raw_z_bitwise_equal':True})
        report['stages'].append(row)
    check_report(report,plan,accepted,result,model,proof,'a'*64);checks.append('report_null_unexecuted_stages_supported')
    bad = copy.deepcopy(report);bad['stages'][1]['metrics'] = {'invented': 0}
    reject('unexecuted_zero_metrics_rejected',lambda:check_report(bad,plan,accepted,result,model,proof,'a'*64))
    bad = copy.deepcopy(report);bad['stages'][0]['metrics'] = {'changed': 2}
    reject('report_cannot_change_saved_metrics',lambda:check_report(bad,plan,accepted,result,model,proof,'a'*64))
    check('historical_attempt_is_copyable_without_source_map_traversal',
          copy_scope('outputs/round13/audit/preflight_attempt.json'))
    rules = copy.deepcopy(plan)
    rules['feature_rules']['alias_mapping']['桃園市'] = '桃園區'
    reject('child_district_alias_not_this_candidate', lambda: rejection_contract(rules,frozen,accepted,result))
    source_names = ['src/round13_alias_inventory.py', 'src/house_features.py', 'src/round6_prepare_geo.py']
    input_names = ['outputs/round6/candidate_plan.json', 'outputs/round6/candidate_plan.sha256',
        'outputs/round6/geography/development/manifest.json', 'outputs/round6/geography/development/manifest.sha256',
        'data_cache/train_features.pkl', 'reports/feature_schema.json',
        'outputs/round6/geography/development/train_source_cache_indices.npy',
        'outputs/round6/models/development/geography_catboost/fit_indices.npy',
        'outputs/round6/models/development/geography_catboost/query_indices.npy',
        'reports/round13_city_alias_feasibility.md', 'reports/round13_official_name_evidence.md']
    rp=copy.deepcopy(plan); rp['old_plan_sha256']='a'*64
    rp['split_contract']['development'].update(fit_indices_content_sha256='b'*64,query_indices_content_sha256='c'*64)
    research={'round':13, 'input_and_source_hashes_unchanged_after_read':True,
        'read_boundaries':{'pure_X_cache_only':True,'development_query_X_read':True,
            'historical_target_containing_dedup_mapping_reused':True,
            **{k:False for k in ('raw_workbooks_read','target_cache_read','labels_predictions_or_scores_read',
                'test_X_read','development_query_is_untouched_blind','model_fit_or_scoring_performed',
                'alias_mapping_inferred_or_applied')}},
        'scope':{'exact_parent_city_values':['桃園縣','桃園市']},
        'phases':{'development_fit':{},'development_query':{}},
        'alignment':{**rp['split_contract']['development'], 'fit_query_overlap_rows':0,
                     'indices_dtype_unique_increasing_range_order_and_dates_verified':True},
        'report':{'path':'reports/round13_alias_inventory.md','sha256':'a'*64},
        'source_sha256':{k:'a'*64 for k in source_names}, 'input_sha256':{k:'a'*64 for k in input_names}}
    check_research(research,rp);checks.append('target_free_inventory_closure_and_canonical_indices_supported')
    for title,section,key,newvalue in [
        ('research_label_use_rejected','read_boundaries','labels_predictions_or_scores_read',True),
        ('research_blind_claim_rejected','read_boundaries','development_query_is_untouched_blind',True),
        ('research_row_mapping_change_rejected','alignment','fit_indices_content_sha256','d'*64)]:
        bad=copy.deepcopy(research);bad[section][key]=newvalue
        reject(title,lambda bad=bad:check_research(bad,rp))
    bad=copy.deepcopy(research);bad['input_sha256'].pop('data_cache/train_features.pkl')
    reject('research_input_omission_rejected',lambda:check_research(bad,rp))
    return {'synthetic_only':True,'real_authorities_labels_models_or_results_read':False,
            'files_written':0,'checks':checks,'count':len(checks),'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create', action='store_true')
    mode.add_argument('--verify', action='store_true')
    mode.add_argument('--self-check', action='store_true')
    args = parser.parse_args()
    if args.self_check:
        print(json.dumps(self_check(), ensure_ascii=False), flush=True)
    elif args.verify:
        print(json.dumps(verify_at(DEST), ensure_ascii=False), flush=True)
    else:
        create()
