"""Archive the finalized R17 development rejection; stdlib and byte hashes only.
Explicit --create / --verify / --self-check. No model or array deserialization,
training, scoring, release, registry write, cleanup, or old-source imports.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round17'
DEST = ROOT / 'runs/round17_experiments'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, CANDIDATE = 'v3_control', 'ordered_geography_blend'
SOURCES = ('src/round17_ordered_geo.py', 'src/round17_experiment.py',
           'src/round17_verify_backtest.py', 'src/round17_report.py',
           'src/archive_round17.py', 'src/round17_x_profile.py')
DOCS = ('reports/round17_plan.md', 'reports/round17_categorical_method_review.md',
        'reports/round17_cold_start_review.md', 'reports/round17_x_profile.md',
        'reports/round17_pipeline_review.md', 'reports/round17_work_log.md',
        'reports/round17_report.md', 'reports/round17_results.json', 'reports/round17_results.sha256')
PROFILE_PATH = 'outputs/round17_research/x_profile.json'
PROFILE_SHA = 'bdfb6e82ccab647a714cbb471218200033e2d8f16dd61b36368fbda697582fd7'
RESEARCH_FILES = (PROFILE_PATH, 'outputs/round17_research/x_profile.sha256')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'predictions.npy', 'raw_z.npy',
               'fit_indices.npy', 'query_indices.npy', 'metadata.json', 'metadata.sha256'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type')
MODEL_SUMMARY_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds') + CAPACITY_FIELDS
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = '''# R17 development-rejection snapshot

This is an experiment archive, not a release. R17 was rejected by its fixed
development gates; confirmation and auxiliary price fits/evaluations were not
performed. The recommendation remains outputs/submission_v3.csv.

The component is raw-price CatBoost: 95 features, 23 categorical features,
2999 iterations and depth 8. Only boosting_type changes from the R6
baseline's Plain to Ordered. Native CBM evidence verifies Ordered boosting,
SymmetricTree and at most 256 leaves per tree.
Create is authorized only after the work log has been closed. No alias
or prefix feature, or LightGBM category-map component, is introduced in R17.
The pure-X profile and its source/receipt are copied as research evidence;
it is not supervised training data or a new price score.

R17 sources, documents, finalized outputs, preparation attempts and logs are
copied byte for byte. Earlier attempts are historical bytes, not declarations
of the current model source closure. Original data, earlier models, all four
annual geographic dependency chains and other declared dependencies remain
external and are checked by SHA-256 and byte length. This is not self-contained.

Verification never deserializes arrays/models or recomputes price errors.
Current originals of the frozen copied files are also checked. versions.json
is only creation-time context; it and CURRENT_STATUS.md are never permanent
external guards. No registry, submission, other experiment file is
written. Partial model/evaluation/report/archive staging is refused and kept.
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
    return (name in SOURCES or (name.startswith('src/round17_') and name.endswith('.py'))
            or name.startswith('outputs/round17/') or name.startswith('outputs/round17_research/')
            or name.startswith('reports/round17_') or name == 'README_round17.md')


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
    need(OUT.is_dir(), 'No R17 outputs')
    for area in ('models', 'evaluation'):
        folder = OUT / area
        need(folder.is_dir() and {p.name for p in folder.iterdir()} == {'development'},
             'Unexpected/incomplete/unauthorized stage in ' + area)
    for name in ('final', 'release'):
        need(not (OUT / name).exists(), 'R17 rejection cannot contain ' + name)
    need(not list((ROOT / 'reports').glob('.round17-report-*')),
         'Partial report staging requires inspection; keep it')
    model = OUT / 'models/development'
    evaluation = OUT / 'evaluation/development'
    need({p.name for p in model.iterdir()} == MODEL_FILES
         and all(p.is_file() for p in model.iterdir()), 'Incomplete model inventory')
    need({p.name for p in evaluation.iterdir()} == {CONTROL + '.npy', CANDIDATE + '.npy',
                                                 'results.json', 'results.sha256'}
         and all(p.is_file() for p in evaluation.iterdir()), 'Incomplete evaluation inventory')
    research_folder = ROOT / 'outputs/round17_research'
    need(research_folder.is_dir(), 'Missing owned pure-X research')
    paths = set(OUT.rglob('*')) | set(research_folder.rglob('*'))
    for path in list((ROOT / 'reports').glob('round17_*')):
        paths.add(path)
        if path.is_dir():
            paths.update(path.rglob('*'))
    paths.update(ROOT / name for name in SOURCES)
    paths.update((ROOT / 'src').glob('round17_*.py'))
    if (ROOT / 'README_round17.md').exists():
        paths.add(ROOT / 'README_round17.md')
    files = set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink is not an archive artifact: ' + str(path))
        if path.is_file():
            name = rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R17 copy scope')
            if name.startswith('reports/'):
                need(not name.lower().endswith(('.tmp', '.part', '.lock')), 'Partial report file')
            need(not (path.name.lower().startswith('submission') and path.suffix.lower() == '.csv'),
                 'A backtest archive must not contain a new submission')
            files.add(name)
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= files, 'Required R17 source/document missing')
    return files


def rejection_contract(plan, frozen, accepted, result):
    need(plan['round'] == frozen['round'] == accepted['round'] == result['round'] == 17,
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
    need(plan['feature_count'] == 95 and plan['categorical_count'] == 23
         and plan['parameters']['iterations'] == 2999 and plan['parameters']['random_seed'] == 42
         and plan['parameters']['depth'] == 8 and plan['parameters']['boosting_type'] == 'Ordered' and plan['parameters']['thread_count'] == 4
         and plan['parameters']['loss_function'] == 'MAE'
         and plan['component'] == 'geography_ordered_raw_catboost'
         and plan['sample_weight'] == 'uniform' and plan['target_mode'] == 'raw',
         'Fixed model contract differs')
    baseline, params = plan['baseline_parameters'], plan['parameters']
    need(baseline['depth'] == 8 and baseline['boosting_type'] == 'Plain' and set(baseline) == set(params)
         and sorted(k for k in params if params[k] != baseline[k]) == ['boosting_type']
         and plan['changed_parameter_names'] == ['boosting_type']
         and plan['model_hyperparameter_change'] == {'boosting_type':['Plain','Ordered']}
         and plan['capacity_constraint'] == {'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Ordered'},
         'Only Plain to Ordered boosting may differ from the R6 baseline')
    need(plan['target_formula'] == 'y_TWD / 1e6'
         and plan['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong target/inverse')
    need(result['fit_rows'] == plan['split_contract']['development']['fit_rows']
         and result['query_rows'] == plan['split_contract']['development']['query_rows'], 'Rows differ')
    for field in ('formulas', 'thresholds'):
        need(result[field] == frozen[field] == plan[field], 'Frozen protocol differs: ' + field)
    need(frozen['parameters'] == plan['parameters'] and result['target_group_weights']
         == frozen['target_group_weights'] == plan['target_group_weights'], 'Frozen parameters/weights differ')


def check_capacity(model, proof):
    capacity = {key:model[key] for key in CAPACITY_FIELDS}
    need(capacity['actual_model_depth'] == 8
         and capacity['actual_boosting_type'] == 'Ordered' and capacity['actual_grow_policy'] == 'SymmetricTree'
         and capacity['actual_trees'] == 2999, 'Wrong native CatBoost capacity')
    need(type(capacity['maximum_tree_leaf_count']) is int
         and 1 <= capacity['maximum_tree_leaf_count'] <= 256, 'Wrong native leaf-cap evidence')
    need(isinstance(capacity['tree_leaf_counts_sha256'],str)
         and re.fullmatch('[0-9a-f]{64}',capacity['tree_leaf_counts_sha256']), 'Invalid LE-int64 leaf digest')
    need(proof['native_model_capacity'] == capacity, 'Proof/model native capacity differs')


def check_report(report, plan, accepted, result, model, proof, proof_sha):
    need(report['round'] == 17 and report['accepted'] is False
         and report['candidate'] == CANDIDATE
         and report['currently_recommended_submission'] == 'outputs/submission_v3.csv'
         and report['stopping_reason'] == accepted['stopping_reason']
         and report['deployment_decision'] == 'keep_v3', 'Report decision differs')
    for key in ('new_submission_created', 'historically_untouched', 'public_score_used', 'private_score_known'):
        need(report[key] is False, 'Report policy differs: ' + key)
    need(report['report_only_reads_saved_JSON_and_hashes_binary_inputs'] is True, 'Report read-scope differs')
    need([item['stage'] for item in report['stages']] == list(STAGES), 'Report stage inventory differs')
    fields = MODEL_SUMMARY_FIELDS
    for item in report['stages']:
        stage = item['stage']; spec = plan['split_contract'][stage]
        need(item['status'] == accepted['stage_status'][stage] and item['dates'] == plan['splits'][stage]
             and item['fit_rows'] == spec['fit_rows'] and item['query_rows'] == spec['query_rows'],
             'Report stage identity/count differs')
        if stage == 'development':
            need(item['metrics'] == result['metrics'] and item['decision'] == result['decision'],
                 'Report changed saved metrics/decision')
            need(item['model'] == {key:model[key] for key in fields}, 'Report model summary differs')
            need(item['independent_review'] == {
                'proof_path': 'outputs/round17/audit/development_replay.json', 'proof_sha256': proof_sha,
                'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal'],
                'native_model_capacity':proof['native_model_capacity']},
                'Report replay proof differs')
        else:
            need(all(item[key] is None for key in ('metrics', 'decision', 'model', 'independent_review')),
                 'Report invents unexecuted scores or models')


def closed_work_log_sha256(raw):
    """Require one exact CLOSED line in the same bytes bound for later copying."""
    need(isinstance(raw, bytes), 'Work log validation requires immutable bytes')
    text = raw.decode('utf-8-sig')
    marker_lines = [line for line in text.splitlines() if 'work_log_status' in line.casefold()]
    need(marker_lines == ['<!-- WORK_LOG_STATUS: CLOSED -->'],
         'Work log must contain exactly one exact CLOSED marker and no other status marker')
    return hashlib.sha256(raw).hexdigest()



def research_mapping(profile, receipt_sha):
    """Expand the fixed descriptive profile without reading its input arrays."""
    need(profile['inputs_and_sources_unchanged'] is True, 'Unverified research input/source')
    need(profile['research_only'] is True and profile['round17_candidate_created'] is False,
         'Saved profile must remain a historical pre-candidate research record')
    access = profile['data_access']
    need(access['only_two_target_free_feature_caches_read'] is True, 'Profile was not target-free')
    for key in ('raw_target_cache_or_Excel_or_valid_read','model_or_prediction_read',
                'price_or_MAE_computed','formal_fit_or_score_performed'):
        need(access[key] is False, 'Unexpected profile data use: '+key)
    need(set(profile['source_sha256']) == {'src/round17_x_profile.py','src/house_features.py'},
         'Unexpected profile source inventory')
    need(set(profile['inputs']) == {'train','test'}, 'Unexpected profile input inventory')
    mapping = {}
    def add(name,digest):
        need(normalized(name) == name and name not in MUTABLE, 'Unsafe/mutable research path')
        need(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest), 'Invalid research digest')
        need(name not in mapping or mapping[name] == digest, 'Conflicting research digest')
        mapping[name] = digest
    add(PROFILE_PATH,PROFILE_SHA)
    add(RESEARCH_FILES[1],receipt_sha)
    for name,digest in profile['source_sha256'].items():add(name,digest)
    for split,item in profile['inputs'].items():
        need(item['path'] == 'data_cache/'+split+'_features.pkl', 'Profile source is not target-free cache')
        need(type(item['bytes']) is int and item['bytes'] >= 0, 'Invalid profile input size')
        add(item['path'],item['sha256'])
    return mapping



def validate_research_map(plan, mapping):
    need(mapping == plan['research_dependency_sha256'], 'Incomplete or expanded profile closure')
    need(all(plan['dependency_sha256'].get(k) == v for k,v in mapping.items()),
         'Research is outside complete frozen dependencies')


def check_research(plan,bindings):
    need(plan['research_dependency_sha256'].get(PROFILE_PATH) == PROFILE_SHA,
         'Plan does not bind the reviewed pure-X profile')
    bindings.add(PROFILE_PATH,PROFILE_SHA)
    profile = bindings.authority(PROFILE_PATH)
    mapping = research_mapping(profile,bindings.values[RESEARCH_FILES[1]]['sha256'])
    validate_research_map(plan, mapping)
    bindings.mapping(mapping)
    for item in profile['inputs'].values():
        need(bindings.values[item['path']]['bytes'] == item['bytes'], 'Pure-X input byte count differs')


def collect():
    """Read finalized authorities and hashes. Never inspect npy/model contents."""
    files = inventory()
    b = Bindings()
    work_log_name = 'reports/round17_work_log.md'
    work_log_bytes = safe(work_log_name).read_bytes()
    b.add(work_log_name, closed_work_log_sha256(work_log_bytes))
    authority_paths = {
        'plan': 'outputs/round17/candidate_plan.json',
        'frozen': 'outputs/round17/frozen_candidate.json',
        'accepted': 'outputs/round17/accepted_candidate.json',
        'result': 'outputs/round17/evaluation/development/results.json',
        'model': 'outputs/round17/models/development/metadata.json',
        'proof': 'outputs/round17/audit/development_replay.json',
        'report': 'reports/round17_results.json',
    }
    values = {key: b.authority(name) for key, name in authority_paths.items()}
    plan, frozen, accepted, result, model, proof, report = (values[k] for k in authority_paths)
    digests = {key: b.values[name]['sha256'] for key, name in authority_paths.items()}
    rejection_contract(plan, frozen, accepted, result)
    for value in (frozen, accepted, result, model, proof):
        need(value['plan_sha256'] == digests['plan'], 'Authority refers to another R17 plan')
    need(accepted['frozen_candidate_sha256'] == digests['frozen']
         and frozen['result_sha256'] == proof['result_sha256'] == digests['result']
         and accepted['stage_result_sha256'] == {'development': digests['result']},
         'Finalized result lineage differs')
    need(proof['frozen_candidate_sha256'] == digests['frozen']
         and proof['accepted_candidate_sha256'] == digests['accepted'], 'Replay final decision identity differs')
    for key in ('plan', 'frozen', 'accepted'):
        field = {'plan': 'plan_sha256', 'frozen': 'frozen_candidate_sha256',
                 'accepted': 'accepted_candidate_sha256'}[key]
        need(report[field] == digests[key], 'Report authority identity differs: ' + field)
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
    check_research(plan, b)
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
    expected_outputs = {'outputs/round17/evaluation/development/' + name + '.npy'
                        for name in (CONTROL, CANDIDATE)}
    need(set(result['output_artifact_sha256']) == expected_outputs, 'Unexpected prediction inventory')
    need(model['round'] == 17 and model['stage'] == 'development'
         and model['component'] == plan['component'] and model['parameters'] == plan['parameters'],
         'Model configuration differs')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['family'] == 'catboost'
         and model['weight_argument_passed_to_pool'] is False
         and model['feature_rules'] == plan['feature_rules']
         and model['feature_count'] == 95 and model['categorical_feature_count'] == 23
         and model['sample_weight'] == 'uniform' and model['target_mode'] == plan['target_mode']
         and model['target_formula'] == plan['target_formula']
         and model['inverse_formula'] == plan['inverse_formula'], 'Model scale/schema differs')
    need(model['baseline_parameters'] == proof['baseline_parameters'] == plan['baseline_parameters']
         and model['changed_parameter_names'] == proof['changed_parameter_names']
         == plan['changed_parameter_names'] == ['boosting_type'], 'Boosting-only recipe evidence differs')
    check_capacity(model, proof)
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
                'calibration_applied', 'extra_price_cap_applied', 'reference_or_trend_applied_here',
                'reference_features_changed'):
        need(model[key] is False, 'Unexpected model behavior: ' + key)
    need(proof['round'] == 17 and proof['stage'] == 'development' and proof['status'] == 'passed'
         and proof['candidate_accepted_in_this_stage'] is False and proof['saved_raw_z_bitwise_equal'] is True,
         'Independent proof does not certify this rejection')
    need(proof['independent_metrics'] == result['metrics']
         and proof['independent_decision'] == result['decision']
         and proof['raw_z_diagnostics'] == model['raw_z_diagnostics'], 'Proof/report payload mismatch')
    for key in ('fit_performed', 'new_window_scored', 'public_score_used', 'historically_untouched'):
        need(proof[key] is False, 'Unexpected independent review behavior: ' + key)
    need(proof['model_family'] == 'catboost' and proof['model_format'] == 'native_CatBoost_cbm'
         and proof['raw_z_units'] == 'million_TWD' and proof['actual_iterations'] == 2999
         and proof['feature_count'] == 95 and proof['categorical_count'] == 23,
         'Independent Ordered CatBoost schema differs')
    for key in ('fit_feature_state_reconstructed_independently',
                'query_model_view_checked_independently',
                'original_95_values_dtypes_order_index_preserved',
                'original_87_reference_and_scoring_keys_unchanged'):
        need(proof[key] is True, 'Independent feature proof differs: ' + key)
    state_evidence = proof['independent_fit_state_evidence']
    need(state_evidence['fit_rows'] == spec['fit_rows']
         and state_evidence['feature_count'] == 95 and state_evidence['categorical_count'] == 23
         and state_evidence['fit_state_reconstructed_independently'] is True
         and state_evidence['fit_model_feature_values_sha256'] == model['fit_model_feature_values_sha256'],
         'Independent fit-only geographic state differs')
    b.add('src/round17_verify_backtest.py', proof['source_sha256'])
    check_report(report, plan, accepted, result, model, proof, digests['proof'])
    # These three source maps are authoritative; verify that the independent
    # replay actually bound the artifacts it says it replayed, without loading them.
    required_proof = {authority_paths[k] for k in ('plan', 'frozen', 'accepted', 'result', 'model')}
    required_proof.update('outputs/round17/models/development/' + name
                          for name in MODEL_FILES)
    required_proof.update(expected_outputs)
    need(required_proof <= set(proof['source_input_sha256']), 'Replay input closure is incomplete')
    required_report = set(required_proof) | {authority_paths['accepted'], authority_paths['proof'],
                                           'src/round17_report.py'}
    need(required_report <= set(report['source_input_sha256']), 'Report input closure is incomplete')
    # Attempts are copied, not interpreted as current source bindings.
    for name in sorted(files):
        b.add(name)
        if name.endswith('.sha256'):
            path = safe(name)
            target = path.with_suffix('.json')
            need(target.is_file(), 'Historical/current receipt has no JSON: ' + name)
            locked(target)  # Byte receipt only; do not traverse historical source maps.
    need(inventory() == files, 'R17 inventory changed during collection')
    need(all(not copy_scope(name) or name in files for name in b.values), 'Owned dependency omitted from copy inventory')
    copied = {name: b.values[name].copy() for name in sorted(files)}
    external = {name: item.copy() for name, item in sorted(b.values.items()) if name not in copied}
    need(not MUTABLE.intersection(set(copied) | set(external)), 'Mutable entry point became permanent')
    registry = safe('runs/versions.json')
    context = {'path': 'runs/versions.json', 'sha256': sha(registry), 'bytes': registry.stat().st_size,
               'plan_time_sha256': plan['registry_sha256_at_plan'],
               'authority_observed_sha256': sorted(b.mutable_declarations.get('runs/versions.json', set())),
               'verification_policy': 'creation-time observation only; never a future external guard'}
    state = {'round': 17, 'is_release': False, 'accepted_backtest': False,
             'candidate': CANDIDATE, 'selected_candidate': CONTROL,
             'stopping_reason': accepted['stopping_reason'], 'stage_status': accepted['stage_status'],
             'recommended_version': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
             'new_submission_created': False, 'registry_modified': False,
             'plan_sha256': digests['plan'], 'frozen_candidate_sha256': digests['frozen'],
             'accepted_candidate_sha256': digests['accepted'], 'development_result_sha256': digests['result'],
             'independent_replay_sha256': digests['proof'], 'report_json_sha256': digests['report'],
             'source_path': 'src/archive_round17.py', 'source_sha256': sha(__file__),
             'property_labels_or_prediction_values_read_by_archiver': False,
             'array_or_model_deserialization_performed': False, 'models_refitted_or_MAE_recomputed': False,
             'index_identity_uses_saved_contract_and_replay_not_array_deserialization': True,
             'historically_untouched': False, 'historical_attempts_kept_without_current_source_enforcement': True,
             'registry_context': context, 'current_status_is_not_bound': True,
             'research_dependency_sha256': plan['research_dependency_sha256'],
             'research_is_target_free_descriptive_evidence': True}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round17_development_rejection_experiments', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 17 and state['is_release'] is False and state['accepted_backtest'] is False
         and state['selected_candidate'] == CONTROL and state['stopping_reason'] == 'development_price_gates_failed'
         and state['recommended_version'] == 'v3' and state['new_submission_created'] is False,
         'Snapshot does not describe the finalized development rejection')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copied and external paths overlap')
    names = set(copied) | set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names} & {n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= set(copied) and all(copy_scope(n) for n in copied),
         'Wrong R17 copy inventory')
    need(all(not copy_scope(n) for n in external), 'Owned R17 artifact left external')
    need(manifest['copied_file_count'] == len(copied)
         and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(item['bytes'] for item in copied.values()),
         'Manifest count/size mismatch')
    for name, item in {**copied, **external}.items():
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid file identity: ' + name)
    for key, name in [('plan_sha256', 'outputs/round17/candidate_plan.json'),
                      ('frozen_candidate_sha256', 'outputs/round17/frozen_candidate.json'),
                      ('accepted_candidate_sha256', 'outputs/round17/accepted_candidate.json'),
                      ('development_result_sha256', 'outputs/round17/evaluation/development/results.json'),
                      ('independent_replay_sha256', 'outputs/round17/audit/development_replay.json'),
                      ('report_json_sha256', 'reports/round17_results.json'),
                      ('source_sha256', 'src/archive_round17.py')]:
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
        need(inventory() == set(copied), 'Original R17 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}


def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round17_experiments_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round17_experiments_staging_' + uuid.uuid4().hex)
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
    manifest = {'kind': 'round17_development_rejection_experiments',
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


def work_log_self_check():
    checks = []
    closed = '<!-- WORK_LOG_STATUS: CLOSED -->'
    for name, raw in [('LF', ('# Log\n'+closed+'\n').encode('utf-8')),
                      ('UTF8_BOM_CRLF', ('\ufeff# 紀錄\r\n'+closed+'\r\n').encode('utf-8'))]:
        need(closed_work_log_sha256(raw) == hashlib.sha256(raw).hexdigest(), 'Work-log bytes SHA differs')
        checks.append('closed_marker_'+name+'_accepted_and_same_bytes_bound')
    cases = {
        'missing': '# Log\n',
        'open': '<!-- WORK_LOG_STATUS: OPEN -->\n',
        'duplicate': closed+'\n'+closed+'\n',
        'closed_plus_open': closed+'\n<!-- WORK_LOG_STATUS: OPEN -->\n',
        'indented': ' '+closed+'\n',
        'unknown_status': closed+'\n<!-- WORK_LOG_STATUS: UNKNOWN -->\n',
        'wrong_case': '<!-- work_log_status: CLOSED -->\n',
    }
    for name,text in cases.items():
        try:
            closed_work_log_sha256(text.encode('utf-8'))
        except RuntimeError:
            checks.append(name+'_rejected')
        else:
            raise AssertionError('Expected work-log rejection: '+name)
    return checks


def self_check():
    """R17 contract deltas only; no authority/model/array reads or archive writes."""
    checks=[]
    def check(name,value):need(bool(value),'Synthetic check failed: '+name);checks.append(name)
    def reject(name,operation):
        try:operation()
        except (RuntimeError,KeyError,TypeError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    baseline={'iterations':2999,'depth':8,'random_seed':42,'thread_count':4,'loss_function':'MAE',
              'boosting_type':'Plain','max_ctr_complexity':2,'one_hot_max_size':8}
    params={**baseline,'boosting_type':'Ordered'}
    plan={'round':17,'candidate':CANDIDATE,'control':CONTROL,'historically_untouched':False,
          'feature_count':95,'categorical_count':23,'parameters':copy.deepcopy(params),
          'baseline_parameters':copy.deepcopy(baseline),'changed_parameter_names':['boosting_type'],
          'capacity_constraint':{'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Ordered'},
          'model_hyperparameter_change':{'boosting_type':['Plain','Ordered']},
          'component':'geography_ordered_raw_catboost','sample_weight':'uniform','target_mode':'raw',
          'target_formula':'y_TWD / 1e6','inverse_formula':'max(z * 1e6, 0)',
          'formulas':{'fixed_synthetic':True},'thresholds':{'fixed_synthetic':True},
          'target_group_weights':{'seen':1.0},'splits':{k:['start','end'] for k in STAGES},
          'split_contract':{k:{'fit_rows':5,'query_rows':3} for k in STAGES}}
    frozen={'round':17,'candidate':CANDIDATE,'passed':False,'later_stages_allowed':False,
            'selected_candidate':CONTROL,'historically_untouched':False,
            **{k:copy.deepcopy(plan[k]) for k in ('formulas','thresholds','parameters','target_group_weights')}}
    accepted={'round':17,'candidate':CANDIDATE,'accepted':False,'selected_candidate':CONTROL,
              'stopping_reason':'development_price_gates_failed','deployment_decision':'keep_v3',
              'stage_status':{k:'price_evaluated' if k=='development' else 'not_executed_after_development_rejection' for k in STAGES},
              'unevaluated_stages':list(STAGES[1:]),'stage_result_sha256':{'development':'a'*64},
              'new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv',
              'no_reselection_or_parameter_change':True,'historically_untouched':False}
    result={'round':17,'stage':'development','evaluated_candidates':[CONTROL,CANDIDATE],
            'decision':{'passed':False,'selected_candidate':CONTROL},'fit_rows':5,'query_rows':3,
            'historically_untouched':False,'metrics':{'synthetic_saved_value':1.0},
            **{k:copy.deepcopy(plan[k]) for k in ('formulas','thresholds','target_group_weights')}}
    rejection_contract(plan,frozen,accepted,result);checks.append('R17_Ordered_only_development_rejection_contract')
    for key,value in [('boosting_type','Plain'),('depth',6),('max_ctr_complexity',1),('one_hot_max_size',16)]:
        bad=copy.deepcopy(plan);bad['parameters'][key]=value
        reject('unplanned_recipe_'+key,lambda bad=bad:rejection_contract(bad,frozen,accepted,result))
    bad=copy.deepcopy(plan);bad['changed_parameter_names']=['depth']
    reject('R15_depth_delta_rejected',lambda:rejection_contract(bad,frozen,accepted,result))
    bad=copy.deepcopy(frozen);bad['passed']=True
    reject('development_pass_not_supported',lambda:rejection_contract(plan,bad,accepted,result))
    model={'family':'catboost','component':plan['component'],'target_mode':'raw',
           'iterations':2999,'actual_iterations':2999,'actual_trees':2999,
           'feature_count':95,'categorical_feature_count':23,'sample_weight':'uniform',
           'weight_argument_passed_to_pool':False,'baseline_parameters':copy.deepcopy(baseline),
           'changed_parameter_names':['boosting_type'],'parameters':copy.deepcopy(params),
           'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_boosting_type':'Ordered',
           'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'b'*64,
           'target_formula':plan['target_formula'],'inverse_formula':plan['inverse_formula'],
           'raw_z_diagnostics':{'synthetic':True},'runner_elapsed_seconds':1.0}
    proof={'replays':{'synthetic':True},'saved_raw_z_bitwise_equal':True,
           'native_model_capacity':{k:model[k] for k in CAPACITY_FIELDS}}
    check_capacity(model,proof);checks.append('native_Ordered_six_fields_match')
    for key,value in [('actual_boosting_type','Plain'),('actual_model_depth',6),('maximum_tree_leaf_count',257)]:
        bad=copy.deepcopy(model);bad[key]=value
        reject('native_'+key+'_mismatch',lambda bad=bad:check_capacity(bad,proof))
    bad=copy.deepcopy(proof);del bad['native_model_capacity']['actual_boosting_type']
    reject('five_field_native_proof_insufficient',lambda:check_capacity(model,bad))
    profile={'inputs_and_sources_unchanged':True,'research_only':True,'round17_candidate_created':False,
             'source_sha256':{'src/round17_x_profile.py':'c'*64,'src/house_features.py':'d'*64},
             'inputs':{k:{'path':'data_cache/'+k+'_features.pkl','bytes':123,'sha256':'e'*64} for k in ('train','test')},
             'data_access':{'only_two_target_free_feature_caches_read':True,'raw_target_cache_or_Excel_or_valid_read':False,
                            'model_or_prediction_read':False,'price_or_MAE_computed':False,'formal_fit_or_score_performed':False}}
    mapping=research_mapping(profile,'a'*64)
    check('all_six_profile_dependencies_expanded',len(mapping)==6 and set(RESEARCH_FILES)<=set(mapping)
          and mapping[PROFILE_PATH]==PROFILE_SHA and 'src/round17_x_profile.py' in mapping)
    research_plan={'research_dependency_sha256':dict(mapping),'dependency_sha256':dict(mapping)}
    validate_research_map(research_plan,mapping);checks.append('research_subset_of_full_frozen_dependencies')
    bad=copy.deepcopy(research_plan);del bad['dependency_sha256']['src/round17_x_profile.py']
    reject('omitted_profile_source_in_full_dependencies',lambda:validate_research_map(bad,mapping))
    bad=copy.deepcopy(research_plan);del bad['research_dependency_sha256']['data_cache/test_features.pkl']
    reject('partial_research_closure',lambda:validate_research_map(bad,mapping))
    bad=copy.deepcopy(profile);bad['inputs']['train']['path']='data_cache/train.pkl'
    reject('target_cache_in_research_rejected',lambda:research_mapping(bad,'a'*64))
    bad=copy.deepcopy(profile);bad['data_access']['price_or_MAE_computed']=True
    reject('research_with_price_scoring_rejected',lambda:research_mapping(bad,'a'*64))
    check('owned_profile_source_JSON_receipt_and_docs_copyable',all(copy_scope(n) for n in SOURCES+DOCS+RESEARCH_FILES)
          and not any(copy_scope(n) for n in ('data_external/geography/taiwan-220101.osm.pbf',
                  'outputs/round6/geography/development/manifest.json','CURRENT_STATUS.md','runs/versions.json',
                  'src/round16_unit_prefix.py','outputs/round16_research/prefix_support.json')))
    report={'round':17,'candidate':CANDIDATE,'accepted':False,
            'currently_recommended_submission':'outputs/submission_v3.csv',
            'stopping_reason':accepted['stopping_reason'],'deployment_decision':'keep_v3',
            'new_submission_created':False,'historically_untouched':False,'public_score_used':False,
            'private_score_known':False,'report_only_reads_saved_JSON_and_hashes_binary_inputs':True,'stages':[]}
    for stage in STAGES:
        row={'stage':stage,'status':accepted['stage_status'][stage],'dates':plan['splits'][stage],
             'fit_rows':5,'query_rows':3,'metrics':None,'decision':None,'model':None,'independent_review':None}
        if stage=='development':
            row.update(metrics=result['metrics'],decision=result['decision'],model={k:model[k] for k in MODEL_SUMMARY_FIELDS},
                independent_review={'proof_path':'outputs/round17/audit/development_replay.json','proof_sha256':'a'*64,
                                    'replays':proof['replays'],'raw_z_bitwise_equal':True,
                                    'native_model_capacity':proof['native_model_capacity']})
        report['stages'].append(row)
    check_report(report,plan,accepted,result,model,proof,'a'*64);checks.append('report_six_capacity_fields_and_unexecuted_nulls_match')
    bad=copy.deepcopy(report);del bad['stages'][0]['model']['actual_boosting_type']
    reject('report_missing_actual_Ordered_evidence',lambda:check_report(bad,plan,accepted,result,model,proof,'a'*64))
    return {'synthetic_only':True,'material_R17_delta_only':True,'inherited_R15_full_suite_rerun':False,
            'real_authorities_labels_models_or_results_read':False,'files_written':0,
            'checks':checks,'count':len(checks),'script_sha256':sha(__file__)}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create',action='store_true')
    mode.add_argument('--verify',action='store_true')
    mode.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    elif args.verify:print(json.dumps(verify_at(DEST),ensure_ascii=False),flush=True)
    else:create()
