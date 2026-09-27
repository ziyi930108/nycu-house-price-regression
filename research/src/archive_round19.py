"""Archive the finalized R19 three-stage later veto; stdlib and byte hashes only.
Explicit --create / --verify / --self-check. No model or array deserialization,
training, scoring, release, registry write, cleanup, or old-source imports.
"""
from __future__ import annotations
import argparse, copy, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round19'
DEST = ROOT / 'runs/round19_experiments'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, CANDIDATE = 'v3_control', 'unit_catboost_geography_blend'
SOURCES = ('src/round19_unit_geo.py', 'src/round19_experiment.py',
           'src/round19_verify_backtest.py', 'src/round19_report.py',
           'src/archive_round19.py', 'src/round19_area_audit.py')
DOCS = ('reports/round19_plan.md', 'reports/round19_area_method_review.md',
        'reports/round19_pipeline_review.md', 'reports/round19_work_log.md',
        'reports/round19_report.md', 'reports/round19_results.json', 'reports/round19_results.sha256')
REMOVED_FEATURES = []
TARGET_FORMULA = 'y_TWD / (effective_area(X) * 1e4)'
INVERSE_FORMULA = 'max(z * (effective_area(X) * 1e4), 0)'
WEIGHT_FORMULA = 'effective_area / 100'
RAW_Z_UNITS = 'ten_thousand_TWD_per_effective_square_metre'
AREA_RULE = 'finite positive building_area; else finite positive parking_area; else 1'
AREA_HASH_FIELDS = ('fit_effective_area_sha256', 'fit_sample_weights_sha256', 'fit_factor_sha256')
RESEARCH_FILES = ('src/round19_area_audit.py', 'outputs/round19_research/area_audit.json',
                  'outputs/round19_research/area_audit.sha256')
RESEARCH_BINDING_FILES = RESEARCH_FILES
MODEL_FILES = {'model.cbm', 'feature_state.json', 'predictions.npy', 'raw_z.npy',
               'fit_indices.npy', 'query_indices.npy', 'metadata.json', 'metadata.sha256'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                   'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type')
MODEL_SUMMARY_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fit_effective_area_sha256', 'fit_sample_weights_sha256', 'fit_factor_sha256',
    'fit_weight_diagnostics', 'native_categorical_feature_count',
    'geo_rule_target_weight_fields_are_provenance_only') + CAPACITY_FIELDS
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = '''# R19 three-stage later-veto snapshot

This is an experiment archive, not a release. Development selected the fixed
R19 candidate and authorized confirmation/auxiliary. All three price backtests
completed, but at least one later gate vetoed deployment. The frozen selected
candidate remains the R19 candidate; the recommendation remains
outputs/submission_v3.csv. No final fit, new release or new CSV was produced.

The component is area-weighted unit-price CatBoost: original R6 95 features,
23 categorical features, 2999 iterations and depth 8. All passed parameters
remain identical, including Plain. Native CBM evidence verifies Plain,
SymmetricTree, at most 256 leaves per tree and 23 categorical positions.
A is finite positive building_area, else finite positive parking_area, else 1.
The Pool target is y/(A*1e4), its weight is A/100, and the TWD prediction is
max(z*A*1e4,0). No extra 1e6 scaling, identity-column removal or new price cap.
Original X87 still serves the unchanged reference/scoring rules.
Create is authorized only after the work log has been closed.
The pure-X area audit source, JSON and its real receipt are copied as research
evidence. Its eight source, one evidence and eleven input declarations merge
with the three seeds into 22 identities; inputs remain externally hash-bound.
No historical attempt, missing receipt or rejected score is manufactured.

R19 sources, documents, finalized outputs, preparation attempts and logs are
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
    return (name in SOURCES or (name.startswith('src/round19_') and name.endswith('.py'))
            or name.startswith('outputs/round19/') or name.startswith('outputs/round19_research/')
            or name.startswith('reports/round19_') or name == 'README_round19.md')


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
    need(OUT.is_dir(), 'No R19 outputs')
    for area in ('models', 'evaluation'):
        folder = OUT / area
        need(folder.is_dir() and {p.name for p in folder.iterdir()} == set(STAGES),
             'Unexpected/incomplete/unauthorized stage in ' + area)
    for name in ('final', 'release'):
        need(not (OUT / name).exists(), 'R19 rejection cannot contain ' + name)
    need(not list((ROOT / 'reports').glob('.round19-report-*')),
         'Partial report staging requires inspection; keep it')
    for stage in STAGES:
        model = OUT / 'models' / stage
        evaluation = OUT / 'evaluation' / stage
        need({p.name for p in model.iterdir()} == MODEL_FILES
             and all(p.is_file() for p in model.iterdir()), 'Incomplete model inventory')
        need({p.name for p in evaluation.iterdir()} == {CONTROL + '.npy', CANDIDATE + '.npy',
                                                     'results.json', 'results.sha256'}
             and all(p.is_file() for p in evaluation.iterdir()), 'Incomplete evaluation inventory')
    research_folder = ROOT / 'outputs/round19_research'
    need(research_folder.is_dir(), 'Missing owned R19 research')
    paths = set(OUT.rglob('*')) | set(research_folder.rglob('*'))
    for path in list((ROOT / 'reports').glob('round19_*')):
        paths.add(path)
        if path.is_dir():
            paths.update(path.rglob('*'))
    paths.update(ROOT / name for name in SOURCES)
    paths.update((ROOT / 'src').glob('round19_*.py'))
    if (ROOT / 'README_round19.md').exists():
        paths.add(ROOT / 'README_round19.md')
    files = set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink is not an archive artifact: ' + str(path))
        if path.is_file():
            name = rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R19 copy scope')
            if name.startswith('reports/'):
                need(not name.lower().endswith(('.tmp', '.part', '.lock')), 'Partial report file')
            need(not (path.name.lower().startswith('submission') and path.suffix.lower() == '.csv'),
                 'A backtest archive must not contain a new submission')
            files.add(name)
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= files, 'Required R19 source/document missing')
    return files


def validate_decision(stage, decision):
    common = {'large_strata_deteriorate_at_most_5pct'}
    if stage == 'development':
        keys = common | {'overall_improves_at_least_1pct', 'weighted_improves_at_least_1pct', 'quarters_deteriorate_at_most_5pct'}
    else:
        need(stage in STAGES[1:], 'Unexpected decision stage')
        keys = common | {'overall_within_limit', 'weighted_within_limit'}
        if stage == 'confirmation': keys.add('at_least_two_quarters_improve')
    need(set(decision['checks']) == keys and all(type(v) is bool for v in decision['checks'].values()), 'Saved gate inventory differs: '+stage)
    need(type(decision['passed']) is bool and decision['passed'] == all(decision['checks'].values()), 'Saved pass/checks differ: '+stage)
    need(decision['selected_candidate'] == (CANDIDATE if decision['passed'] else CONTROL)
         and decision['candidate_reselection_performed'] is False, 'Saved stage selection differs')
    quarters=decision['improved_quarters']
    need(isinstance(quarters,list) and len(set(quarters))==len(quarters)
         and set(quarters)<={'Q1','Q2','Q3','Q4'}, 'Invalid improved-quarter inventory')


def finalized_contract(plan, frozen, accepted, results):
    need(set(results) == set(STAGES), 'All three completed stages are required; missing work is not veto')
    need(plan['round'] == frozen['round'] == accepted['round'] == 19, 'Wrong experiment round')
    need(plan['candidate'] == frozen['candidate'] == accepted['candidate'] == CANDIDATE
         and plan['control'] == CONTROL, 'Wrong candidate/control')
    need(frozen['passed'] is True and frozen['later_stages_allowed'] is True
         and frozen['selected_candidate'] == CANDIDATE, 'Expected authorized later stages after development pass')
    need(accepted['accepted'] is False and accepted['selected_candidate'] == CANDIDATE
         and accepted['stopping_reason'] == 'later_backtest_gates_failed'
         and accepted['deployment_decision'] == 'keep_v3', 'Only finalized later veto is supported')
    need(accepted['stage_status'] == dict.fromkeys(STAGES,'price_evaluated')
         and accepted['unevaluated_stages'] == []
         and set(accepted['stage_result_sha256']) == set(STAGES), 'Wrong completed-stage inventory')
    need(accepted['new_submission_created'] is False
         and accepted['currently_recommended_submission'] == 'outputs/submission_v3.csv'
         and accepted['no_reselection_or_parameter_change'] is True
         and accepted['private_score_known'] is False, 'Unexpected release, score or reselection')
    for value in (plan, frozen, accepted, *results.values()):
        need(value['historically_untouched'] is False, 'Historical reuse must remain explicit')
    need(plan['feature_count'] == 95 and plan['categorical_count'] == 23
         and plan['parameters']['iterations'] == 2999 and plan['parameters']['random_seed'] == 42
         and plan['parameters']['depth'] == 8 and plan['parameters']['boosting_type'] == 'Plain' and plan['parameters']['thread_count'] == 4
         and plan['parameters']['loss_function'] == 'MAE'
         and plan['component'] == 'geography_unit_catboost'
         and plan['sample_weight'] == WEIGHT_FORMULA and plan['target_mode'] == 'unit'
         and plan['weight_argument_passed_to_pool'] is True,
         'Fixed model contract differs')
    baseline, params = plan['baseline_parameters'], plan['parameters']
    need(baseline == params and baseline['depth'] == 8 and baseline['boosting_type'] == 'Plain'
         and plan['changed_parameter_names'] == [] and plan['model_hyperparameter_change'] == {}
         and plan['capacity_constraint'] == {'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'},
         'All parameters must equal the original R6 baseline')
    need(plan['removed_feature_names'] == REMOVED_FEATURES
         and plan['feature_rules']['removed_feature_names'] == REMOVED_FEATURES,
         'Original model features must remain complete')
    need(plan['target_formula'] == TARGET_FORMULA and plan['inverse_formula'] == INVERSE_FORMULA
         and plan['model_target_change'] == {'from':'raw','to':'unit'}
         and plan['sample_weight_formula'] == WEIGHT_FORMULA, 'Wrong unit target/inverse/weight')

    for stage, result in results.items():
        need(result['round']==19 and result['stage']==stage
             and result['evaluated_candidates']==[CONTROL,CANDIDATE], 'Wrong evaluated stage/candidates')
        validate_decision(stage,result['decision'])
        spec=plan['split_contract'][stage]
        need(result['fit_rows']==spec['fit_rows'] and result['query_rows']==spec['query_rows'], 'Rows differ')
        for field in ('formulas','thresholds','target_group_weights'):
            need(result[field]==frozen[field]==plan[field], 'Frozen protocol differs: '+field)
    need(results['development']['decision']['passed'] is True
         and any(results[stage]['decision']['passed'] is False for stage in STAGES[1:]),
         'Completed decisions do not support development pass followed by later veto')
    need(frozen['parameters']==plan['parameters'], 'Frozen parameters differ')


def check_capacity(model, proof):
    capacity = {key:model[key] for key in CAPACITY_FIELDS}
    need(capacity['actual_model_depth'] == 8
         and capacity['actual_boosting_type'] == 'Plain' and capacity['actual_grow_policy'] == 'SymmetricTree'
         and capacity['actual_trees'] == 2999, 'Wrong native CatBoost capacity')
    need(type(capacity['maximum_tree_leaf_count']) is int
         and 1 <= capacity['maximum_tree_leaf_count'] <= 256, 'Wrong native leaf-cap evidence')
    need(isinstance(capacity['tree_leaf_counts_sha256'],str)
         and re.fullmatch('[0-9a-f]{64}',capacity['tree_leaf_counts_sha256']), 'Invalid LE-int64 leaf digest')
    need(proof['native_model_capacity'] == capacity, 'Proof/model native capacity differs')


def check_report(report, plan, accepted, results, models, proofs, proof_shas):
    need(report['round'] == 19 and report['accepted'] is False
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
        result, model, proof, proof_sha = results[stage], models[stage], proofs[stage], proof_shas[stage]
        need(item['metrics'] == result['metrics'] and item['decision'] == result['decision'],
             'Report changed saved metrics/decision')
        need(item['model'] == {key:model[key] for key in fields}, 'Report model summary differs')
        need(item['independent_review'] == {
            'proof_path': 'outputs/round19/audit/'+stage+'_replay.json', 'proof_sha256': proof_sha,
            'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal'],
            'native_model_capacity':proof['native_model_capacity'],
            'removed_feature_names':proof['removed_feature_names'],
            'independent_fit_state_evidence':proof['independent_fit_state_evidence'],
            'independent_area_weight_evidence':proof['independent_area_weight_evidence']},
            'Report replay proof differs')

def closed_work_log_sha256(raw):
    """Require one exact CLOSED line in the same bytes bound for later copying."""
    need(isinstance(raw, bytes), 'Work log validation requires immutable bytes')
    text = raw.decode('utf-8-sig')
    marker_lines = [line for line in text.splitlines() if 'work_log_status' in line.casefold()]
    need(marker_lines == ['<!-- WORK_LOG_STATUS: CLOSED -->'],
         'Work log must contain exactly one exact CLOSED marker and no other status marker')
    return hashlib.sha256(raw).hexdigest()



def research_mapping(record, initial_mapping):
    """Expand the actual pure-X area audit and its real receipt; never decode inputs."""
    need(record['round']==19 and record['status']=='target_free_effective_area_support_audit'
         and record['feature_values_modified'] is False and record['prices_None_R1_transform_used'] is True,
         'Wrong area research record')
    scope=record['read_scope']
    need(scope['target_cells_decoded']==0 and scope['train_test_target_free_caches_only'] is True
         and scope['query_test_weights_descriptive_not_training'] is True, 'Area research scope differs')
    for key in ('MAE_computed','candidate_fitted','model_or_prediction_loaded','raw_price_cache_or_Excel_loaded','valid_dataset_loaded'):
        need(scope[key] is False, 'Unexpected area research use: '+key)
    alignment=record['alignment']
    need(alignment['canonical_rows']==455255 and alignment['fit_rows']==240316 and alignment['query_rows']==81234
         and alignment['indices_and_dates_match_released_R6'] is True and alignment['original_saved_mapping_reused'] is True
         and alignment['target_dedup_recomputed_here'] is False, 'Area research mapping declaration differs')
    need(set(initial_mapping)==set(RESEARCH_BINDING_FILES) and len(record['source_sha256'])==8
         and len(record['evidence_sha256'])==1 and len(record['inputs'])==11, 'Research declaration inventory differs')
    mapping={}
    def add(name,digest):
        need(normalized(name)==name and name not in MUTABLE, 'Unsafe/mutable research path')
        need(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest), 'Invalid research digest')
        need(name not in mapping or mapping[name]==digest, 'Conflicting research digest')
        mapping[name]=digest
    for name,digest in initial_mapping.items():add(name,digest)
    for field in ('source_sha256','evidence_sha256'):
        for name,digest in record[field].items():add(name,digest)
    for name,item in record['inputs'].items():
        need(name==item['path'] and type(item['bytes']) is int and item['bytes']>=0, 'Research input path/size differs')
        add(name,item['sha256'])
    need(len(mapping)==22, 'Expected the exact merged 22 area-research dependencies')
    return mapping


def validate_research_map(plan, mapping):
    need(mapping == plan['research_dependency_sha256'], 'Incomplete or expanded research closure')
    need(all(plan['dependency_sha256'].get(k) == v for k,v in mapping.items()),
         'Research is outside complete frozen dependencies')


def check_research(plan, bindings):
    initial={}
    for name in RESEARCH_BINDING_FILES:
        expected=plan['research_dependency_sha256'][name]
        bindings.add(name,expected);initial[name]=expected
    record=bindings.authority(RESEARCH_FILES[1])
    mapping=research_mapping(record,initial)
    validate_research_map(plan,mapping);bindings.mapping(mapping)
    for item in record['inputs'].values():
        need(bindings.values[item['path']]['bytes']==item['bytes'], 'Area research input byte length differs')
    return {'frozen_research_dependency_count':len(mapping),'dependency_sha256':copy.deepcopy(mapping),
            'read_scope':copy.deepcopy(record['read_scope']),'alignment':copy.deepcopy(record['alignment']),
            'research_summary_not_used_as_backtest_fit_rows':True,
            'report_did_not_load_research_arrays_or_recompute_statistics':True}


def check_area_evidence(model, proof, feature_state):
    """Check JSON state, model metadata and independent unit evidence without arrays."""
    content={k:v for k,v in feature_state.items() if k!='state_content_sha256'}
    digest=hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
    need(feature_state['state_content_sha256']==model['feature_state_content_sha256']==digest,
         'Saved feature-state content identity differs')
    need(feature_state['source_sha256']==model['component_source_sha256']
         and feature_state['parameters']==model['parameters']==model['baseline_parameters']
         and feature_state['removed_feature_names']==model['removed_feature_names']==[]
         and feature_state['changed_parameter_names']==model['changed_parameter_names']==[], 'Feature-state recipe differs')
    need(model['target_mode']==feature_state['target_mode']=='unit'
         and model['sample_weight']==WEIGHT_FORMULA and model['weight_argument_passed_to_pool'] is True
         and model['native_categorical_feature_count']==23
         and model['geo_rule_target_weight_fields_are_provenance_only'] is True, 'Wrong unit Pool/native category contract')
    area=proof['independent_area_weight_evidence']
    need(area==proof['independent_fit_state_evidence']['independent_area_weight_evidence'],
         'Top-level/nested independent area evidence differs')
    for key in (*AREA_HASH_FIELDS,'fit_weight_diagnostics','target_formula','inverse_formula'):
        need(area[key]==model[key]==feature_state[key], 'Fit area/weight/state evidence differs: '+key)
    for key in AREA_HASH_FIELDS:
        need(re.fullmatch('[0-9a-f]{64}',model[key]) is not None, 'Invalid area/weight/factor digest')
    need(area['target_formula']==TARGET_FORMULA and area['inverse_formula']==INVERSE_FORMULA
         and area['sample_weight']==WEIGHT_FORMULA and area['raw_z_units']==RAW_Z_UNITS
         and proof['raw_z_units']==RAW_Z_UNITS, 'Unit-scale evidence differs')
    for key in ('area_weight_factor_recomputed_from_original_X',
                'Pool_weight_usage_verified_from_frozen_source_and_saved_metadata','historical_fit_not_reexecuted'):
        need(area[key] is True, 'Independent area evidence scope differs')
    d=model['fit_weight_diagnostics'];counts=d['effective_area_source_counts']
    need(d['rows']==model['fit_rows'] and d['effective_area_rule']==AREA_RULE
         and d['weights_normalized_to_mean_one'] is False
         and set(counts)=={'building_area','parking_area_fallback','default_one_fallback'}
         and all(type(v) is int and v>=0 for v in counts.values()) and sum(counts.values())==model['fit_rows'],
         'Saved fit-only weight diagnostic differs')
    need(len(feature_state['all_feature_names'])==len(feature_state['geo_state']['all_feature_names'])==95
         and feature_state['all_feature_names']==feature_state['geo_state']['all_feature_names']
         and len(feature_state['categorical_columns'])==23, 'Unit feature state does not preserve original95/23')


def check_stage(stage, plan, result, model, proof, b, authority_paths, digests):
    """Validate the fixed stage's saved JSON and bytes; never load its arrays/model."""
    for value in (result,model,proof):
        need(value['plan_sha256']==digests['plan'], 'Stage refers to another R19 plan')
    for value in (result,model):
        need(value['source_sha256']==plan['source_sha256'], 'Stage source closure differs')
        need(value['frozen_candidate_sha256']==(None if stage=='development' else digests['frozen']),
             'Stage development authorization differs')
    need(proof['result_sha256']==digests['result']
         and proof['frozen_candidate_sha256']==digests['frozen']
         and proof['accepted_candidate_sha256']==digests['accepted'], 'Replay finalized lineage differs')
    for field in ('input_artifact_sha256','output_artifact_sha256'):
        b.mapping(result[field])
    b.mapping(proof['source_input_sha256'])
    b.mapping(proof['protected_artifacts_unchanged'])
    b.mapping(model['component_source_sha256'])
    mf = OUT / 'models' / stage
    need(set(model['artifact_sha256']) == MODEL_FILES - {'metadata.json', 'metadata.sha256'},
         'Model metadata inventory differs')
    for name, digest in model['artifact_sha256'].items():
        need(Path(name).name == name, 'Unsafe model artifact name')
        b.add(rel(mf / name), digest)
    expected_outputs = {'outputs/round19/evaluation/' + stage + '/' + name + '.npy'
                        for name in (CONTROL, CANDIDATE)}
    need(set(result['output_artifact_sha256']) == expected_outputs, 'Unexpected prediction inventory')
    need(model['round'] == 19 and model['stage'] == stage
         and model['component'] == plan['component'] and model['parameters'] == plan['parameters'],
         'Model configuration differs')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['family'] == 'catboost'
         and model['weight_argument_passed_to_pool'] is True
         and model['feature_rules'] == plan['feature_rules']
         and model['feature_count'] == 95 and model['categorical_feature_count'] == 23
         and model['sample_weight'] == WEIGHT_FORMULA and model['target_mode'] == plan['target_mode']
         and model['target_formula'] == plan['target_formula']
         and model['inverse_formula'] == plan['inverse_formula'], 'Model scale/schema differs')
    need(model['baseline_parameters'] == proof['baseline_parameters'] == plan['baseline_parameters']
         and model['changed_parameter_names'] == proof['changed_parameter_names']
         == plan['changed_parameter_names'] == []
         and model['parameters'] == model['baseline_parameters']
         and model['removed_feature_names'] == proof['removed_feature_names'] == plan['removed_feature_names'] == REMOVED_FEATURES,
         'Unchanged recipe or fixed feature-removal evidence differs')
    check_capacity(model, proof)
    spec = plan['split_contract'][stage]
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
    need(proof['round'] == 19 and proof['stage'] == stage and proof['status'] == 'passed'
         and proof['candidate_accepted_in_this_stage'] is result['decision']['passed'] and proof['saved_raw_z_bitwise_equal'] is True,
         'Independent proof does not certify this stage')
    need(proof['independent_metrics'] == result['metrics']
         and proof['independent_decision'] == result['decision']
         and proof['raw_z_diagnostics'] == model['raw_z_diagnostics'], 'Proof/report payload mismatch')
    for key in ('fit_performed', 'new_window_scored', 'public_score_used', 'historically_untouched'):
        need(proof[key] is False, 'Unexpected independent review behavior: ' + key)
    need(proof['model_family'] == 'catboost' and proof['model_format'] == 'native_CatBoost_cbm'
         and proof['raw_z_units'] == RAW_Z_UNITS and proof['actual_iterations'] == 2999
         and proof['feature_count'] == 95 and proof['categorical_count'] == 23,
         'Independent unit CatBoost schema differs')
    for key in ('fit_feature_state_reconstructed_independently',
                'query_model_view_checked_independently',
                'original_95_values_dtypes_order_index_preserved',
                'original_87_reference_and_scoring_keys_unchanged'):
        need(proof[key] is True, 'Independent feature proof differs: ' + key)
    state_evidence = proof['independent_fit_state_evidence']
    need(state_evidence['fit_rows'] == spec['fit_rows']
         and state_evidence['base_feature_count'] == 95
         and state_evidence['feature_count'] == 95 and state_evidence['categorical_count'] == 23
         and state_evidence['removed_feature_names'] == REMOVED_FEATURES
         and state_evidence['fit_state_reconstructed_independently'] is True
         and state_evidence['fit_model_feature_values_sha256'] == model['fit_model_feature_values_sha256'],
         'Independent fit-only geographic state differs')
    feature_state=read(safe('outputs/round19/models/' + stage + '/feature_state.json'))
    check_area_evidence(model,proof,feature_state)
    b.add('src/round19_verify_backtest.py', proof['source_sha256'])
    # These three source maps are authoritative; verify that the independent
    # replay actually bound the artifacts it says it replayed, without loading them.
    required_proof = {authority_paths[k] for k in ('plan', 'frozen', 'accepted', 'result', 'model')}
    required_proof.update('outputs/round19/models/' + stage + '/' + name
                          for name in MODEL_FILES)
    required_proof.update(expected_outputs)
    need(required_proof <= set(proof['source_input_sha256']), 'Replay input closure is incomplete')
    required_report = set(required_proof) | {authority_paths['accepted'], authority_paths['proof'],
                                           'src/round19_report.py'}
    return required_report


def collect():
    """Read finalized authorities and hashes. Never inspect npy/model contents."""
    files = inventory()
    b = Bindings()
    work_log_name = 'reports/round19_work_log.md'
    work_log_bytes = safe(work_log_name).read_bytes()
    b.add(work_log_name, closed_work_log_sha256(work_log_bytes))
    authority_paths = {
        'plan':'outputs/round19/candidate_plan.json',
        'frozen':'outputs/round19/frozen_candidate.json',
        'accepted':'outputs/round19/accepted_candidate.json',
        'report':'reports/round19_results.json',
    }
    values={key:b.authority(name) for key,name in authority_paths.items()}
    plan,frozen,accepted,report=(values[key] for key in authority_paths)
    digests={key:b.values[name]['sha256'] for key,name in authority_paths.items()}
    results,models,proofs,stage_paths,stage_digests={},{},{},{},{}
    for stage in STAGES:
        paths={**authority_paths,'result':'outputs/round19/evaluation/'+stage+'/results.json',
               'model':'outputs/round19/models/'+stage+'/metadata.json',
               'proof':'outputs/round19/audit/'+stage+'_replay.json'}
        results[stage]=b.authority(paths['result'])
        models[stage]=b.authority(paths['model'])
        proofs[stage]=b.authority(paths['proof'])
        stage_paths[stage]=paths
        stage_digests[stage]={key:b.values[name]['sha256'] for key,name in paths.items()}
    finalized_contract(plan,frozen,accepted,results)
    for value in (frozen,accepted):
        need(value['plan_sha256']==digests['plan'] and value['source_sha256']==plan['source_sha256'],
             'Final decision plan/source differs')
    need(accepted['frozen_candidate_sha256']==digests['frozen']
         and frozen['result_sha256']==stage_digests['development']['result']
         and accepted['stage_result_sha256']=={stage:stage_digests[stage]['result'] for stage in STAGES},
         'Finalized result lineage differs')
    for key,field in [('plan','plan_sha256'),('frozen','frozen_candidate_sha256'),('accepted','accepted_candidate_sha256')]:
        need(report[field]==digests[key], 'Report authority identity differs: '+field)
    for field in ('source_sha256','dependency_sha256','final_dependency_sha256'):
        b.mapping(plan[field])
    need(set(plan['stage_dependency_sha256'])==set(STAGES), 'Incomplete annual dependency maps')
    for mapping in plan['stage_dependency_sha256'].values():b.mapping(mapping)
    b.mapping(frozen['development_artifact_sha256'])
    expected_development={'outputs/round19/models/development/'+name for name in MODEL_FILES}
    expected_development.update('outputs/round19/evaluation/development/'+name for name in
                                ('results.json','results.sha256',CONTROL+'.npy',CANDIDATE+'.npy'))
    need(set(frozen['development_artifact_sha256'])==expected_development, 'Frozen development inventory differs')
    b.mapping(report['source_input_sha256'])
    research_summary=check_research(plan,b)
    need(report['research_summary']==research_summary, 'Report research scope differs')
    b.add('src/round19_report.py',report['source_input_sha256']['src/round19_report.py'])
    # All four geography chains are already plan-bound, including the unused final
    # map. Assert the expansion is complete; never silently drop an edge.
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
    for stage in STAGES:
        required_report=check_stage(stage,plan,results[stage],models[stage],proofs[stage],b,
                                    stage_paths[stage],stage_digests[stage])
        need(required_report<=set(report['source_input_sha256']), 'Report input closure is incomplete: '+stage)
    check_report(report,plan,accepted,results,models,proofs,
                 {stage:stage_digests[stage]['proof'] for stage in STAGES})
    # Attempts are copied, not interpreted as current source bindings.
    for name in sorted(files):
        b.add(name)
        if name.endswith('.sha256'):
            path = safe(name)
            target = path.with_suffix('.json')
            need(target.is_file(), 'Historical/current receipt has no JSON: ' + name)
            locked(target)  # Byte receipt only; do not traverse historical source maps.
    need(inventory() == files, 'R19 inventory changed during collection')
    need(all(not copy_scope(name) or name in files for name in b.values), 'Owned dependency omitted from copy inventory')
    copied = {name: b.values[name].copy() for name in sorted(files)}
    external = {name: item.copy() for name, item in sorted(b.values.items()) if name not in copied}
    need(not MUTABLE.intersection(set(copied) | set(external)), 'Mutable entry point became permanent')
    registry = safe('runs/versions.json')
    need(sha(registry)==plan['registry_sha256_at_plan']
         and read(registry)['recommended_version']=='v3', 'Registry changed before experiment archive')
    context = {'path': 'runs/versions.json', 'sha256': sha(registry), 'bytes': registry.stat().st_size,
               'plan_time_sha256': plan['registry_sha256_at_plan'],
               'authority_observed_sha256': sorted(b.mutable_declarations.get('runs/versions.json', set())),
               'verification_policy': 'creation-time observation only; never a future external guard'}
    state = {'round': 19, 'is_release': False, 'accepted_backtest': False,
             'candidate': CANDIDATE, 'selected_candidate': CANDIDATE,
             'stopping_reason': accepted['stopping_reason'], 'stage_status': accepted['stage_status'],
             'recommended_version': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
             'new_submission_created': False, 'registry_modified': False,
             'plan_sha256': digests['plan'], 'frozen_candidate_sha256': digests['frozen'],
             'accepted_candidate_sha256': digests['accepted'],
             'stage_result_sha256': {stage:stage_digests[stage]['result'] for stage in STAGES},
             'stage_model_metadata_sha256': {stage:stage_digests[stage]['model'] for stage in STAGES},
             'stage_replay_sha256': {stage:stage_digests[stage]['proof'] for stage in STAGES},
             'completed_stages': list(STAGES), 'unevaluated_stages': [],
             'report_json_sha256': digests['report'],
             'source_path': 'src/archive_round19.py', 'source_sha256': sha(__file__),
             'property_labels_or_prediction_values_read_by_archiver': False,
             'array_or_model_deserialization_performed': False, 'models_refitted_or_MAE_recomputed': False,
             'index_identity_uses_saved_contract_and_replay_not_array_deserialization': True,
             'historically_untouched': False, 'historical_attempts_kept_without_current_source_enforcement': True,
             'registry_context': context, 'current_status_is_not_bound': True,
             'research_dependency_sha256': plan['research_dependency_sha256'],
             'research_is_target_free_descriptive_evidence': True,
             'research_area_audit_has_real_receipt': True, 'research_summary': research_summary}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round19_later_veto_experiments', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 19 and state['is_release'] is False and state['accepted_backtest'] is False
         and state['selected_candidate'] == CANDIDATE and state['stopping_reason'] == 'later_backtest_gates_failed'
         and state['completed_stages']==list(STAGES) and state['unevaluated_stages']==[]
         and state['stage_status']==dict.fromkeys(STAGES,'price_evaluated')
         and state['recommended_version'] == 'v3' and state['new_submission_created'] is False,
         'Snapshot does not describe the finalized three-stage later veto')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copied and external paths overlap')
    names = set(copied) | set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names} & {n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= set(copied) and all(copy_scope(n) for n in copied),
         'Wrong R19 copy inventory')
    need(all(not copy_scope(n) for n in external), 'Owned R19 artifact left external')
    need(manifest['copied_file_count'] == len(copied)
         and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(item['bytes'] for item in copied.values()),
         'Manifest count/size mismatch')
    for name, item in {**copied, **external}.items():
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid file identity: ' + name)
    for key, name in [('plan_sha256', 'outputs/round19/candidate_plan.json'),
                      ('frozen_candidate_sha256', 'outputs/round19/frozen_candidate.json'),
                      ('accepted_candidate_sha256', 'outputs/round19/accepted_candidate.json'),
                      ('report_json_sha256', 'reports/round19_results.json'),
                      ('source_sha256', 'src/archive_round19.py')]:
        need(state[key] == copied[name]['sha256'], 'Manifest authority identity mismatch: ' + key)
    for key,pattern in [('stage_result_sha256','outputs/round19/evaluation/{stage}/results.json'),
                        ('stage_model_metadata_sha256','outputs/round19/models/{stage}/metadata.json'),
                        ('stage_replay_sha256','outputs/round19/audit/{stage}_replay.json')]:
        need(set(state[key])==set(STAGES), 'Incomplete stage SHA inventory: '+key)
        for stage in STAGES:
            need(state[key][stage]==copied[pattern.format(stage=stage)]['sha256'], 'Stage authority SHA differs: '+key+'/'+stage)
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
        need(inventory() == set(copied), 'Original R19 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}


def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round19_experiments_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round19_experiments_staging_' + uuid.uuid4().hex)
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
    manifest = {'kind': 'round19_later_veto_experiments',
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
    """Complete bounded archive contract, source-scope and CLOSED synthetic checks."""
    checks=list(work_log_self_check())
    def check(name,value):need(bool(value),'Synthetic check failed: '+name);checks.append(name)
    def reject(name,operation):
        try:operation()
        except (RuntimeError,KeyError,TypeError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    params={'iterations':2999,'depth':8,'random_seed':42,'thread_count':4,'loss_function':'MAE','boosting_type':'Plain'}
    plan={'round':19,'candidate':CANDIDATE,'control':CONTROL,'historically_untouched':False,'feature_count':95,'categorical_count':23,
          'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(params),'changed_parameter_names':[],
          'removed_feature_names':[],'feature_rules':{'removed_feature_names':[]},
          'capacity_constraint':{'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'},
          'model_hyperparameter_change':{},'component':'geography_unit_catboost','sample_weight':WEIGHT_FORMULA,
          'target_mode':'unit','target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
          'weight_argument_passed_to_pool':True,'model_target_change':{'from':'raw','to':'unit'},'sample_weight_formula':WEIGHT_FORMULA,
          'formulas':{'synthetic':True},'thresholds':{'synthetic':True},'target_group_weights':{'seen':1.0},
          'splits':{k:['start','end'] for k in STAGES},'split_contract':{k:{'fit_rows':5,'query_rows':3} for k in STAGES}}
    def decision(stage,passed):
        keys={'large_strata_deteriorate_at_most_5pct'}
        if stage=='development':keys.update(('overall_improves_at_least_1pct','weighted_improves_at_least_1pct','quarters_deteriorate_at_most_5pct'))
        else:
            keys.update(('overall_within_limit','weighted_within_limit'))
            if stage=='confirmation':keys.add('at_least_two_quarters_improve')
        gates=dict.fromkeys(sorted(keys),True)
        if not passed:gates['overall_within_limit' if stage!='development' else 'overall_improves_at_least_1pct']=False
        return {'passed':passed,'selected_candidate':CANDIDATE if passed else CONTROL,'checks':gates,
                'improved_quarters':['Q1','Q4'],'candidate_reselection_performed':False}
    frozen={'round':19,'candidate':CANDIDATE,'passed':True,'later_stages_allowed':True,'selected_candidate':CANDIDATE,
            'historically_untouched':False,**{k:copy.deepcopy(plan[k]) for k in ('formulas','thresholds','parameters','target_group_weights')}}
    accepted={'round':19,'candidate':CANDIDATE,'accepted':False,'selected_candidate':CANDIDATE,
              'stopping_reason':'later_backtest_gates_failed','deployment_decision':'keep_v3',
              'stage_status':dict.fromkeys(STAGES,'price_evaluated'),
              'unevaluated_stages':[],'stage_result_sha256':dict.fromkeys(STAGES,'a'*64),
              'new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv',
              'no_reselection_or_parameter_change':True,'historically_untouched':False,'private_score_known':False}
    results={stage:{'round':19,'stage':stage,'evaluated_candidates':[CONTROL,CANDIDATE],
             'decision':decision(stage,stage!='confirmation'),'fit_rows':5,'query_rows':3,'historically_untouched':False,
             'metrics':{'synthetic_stage':stage},**{k:copy.deepcopy(plan[k]) for k in ('formulas','thresholds','target_group_weights')}}
             for stage in STAGES}
    finalized_contract(plan,frozen,accepted,results);checks.append('unit95_23_three_completed_stages_confirmation_veto')
    for key,value in [('feature_count',91),('categorical_count',19),('target_mode','raw'),
                      ('sample_weight','uniform'),('removed_feature_names',['project_name']),('weight_argument_passed_to_pool',False)]:
        bad=copy.deepcopy(plan);bad[key]=value
        reject('wrong_plan_'+key,lambda bad=bad:finalized_contract(bad,frozen,accepted,results))
    for key,value in [('passed',False),('later_stages_allowed',False),('selected_candidate',CONTROL)]:
        bad=copy.deepcopy(frozen);bad[key]=value
        reject('wrong_freeze_'+key,lambda bad=bad:finalized_contract(plan,bad,accepted,results))
    for key,value in [('accepted',True),('selected_candidate',CONTROL),('deployment_decision',CANDIDATE),
                      ('stopping_reason','development_price_gates_failed'),('unevaluated_stages',['auxiliary'])]:
        bad=copy.deepcopy(accepted);bad[key]=value
        reject('wrong_acceptance_'+key,lambda bad=bad:finalized_contract(plan,frozen,bad,results))
    bad=copy.deepcopy(results);bad.pop('auxiliary')
    reject('missing_auxiliary_is_incomplete_not_rejection',lambda:finalized_contract(plan,frozen,accepted,bad))
    bad=copy.deepcopy(results);bad['development']['decision']=decision('development',False)
    reject('development_rejection_not_three_stage_veto',lambda:finalized_contract(plan,frozen,accepted,bad))
    bad=copy.deepcopy(results);bad['confirmation']['decision']=decision('confirmation',True)
    reject('all_passed_requires_different_archive_scope',lambda:finalized_contract(plan,frozen,accepted,bad))
    bad=copy.deepcopy(results);bad['confirmation']['decision']=decision('confirmation',True);bad['auxiliary']['decision']=decision('auxiliary',False)
    finalized_contract(plan,frozen,accepted,bad);checks.append('auxiliary_veto_supported_without_reselection')
    bad=copy.deepcopy(results);bad['auxiliary']['decision']=decision('auxiliary',False)
    finalized_contract(plan,frozen,accepted,bad);checks.append('both_later_vetoes_supported')
    bad=copy.deepcopy(results);bad['confirmation']['decision']['passed']=True
    reject('saved_gate_boolean_inconsistency_rejected',lambda:finalized_contract(plan,frozen,accepted,bad))
    bad=copy.deepcopy(results);bad['auxiliary']['decision']['checks']['quarters_deteriorate_at_most_5pct']=True
    reject('auxiliary_extra_gate_rejected',lambda:finalized_contract(plan,frozen,accepted,bad))
    diag={'rows':5,'effective_area_rule':AREA_RULE,'weights_normalized_to_mean_one':False,
          'effective_area_source_counts':{'building_area':3,'parking_area_fallback':1,'default_one_fallback':1},
          'effective_area':{'min':1.,'max':100.,'mean':20.},'sample_weight':{'min':.01,'max':1.,'mean':.2},
          'price_factor':{'min':1e4,'max':1e6,'mean':2e5}}
    model={'family':'catboost','component':plan['component'],'target_mode':'unit','iterations':2999,'actual_iterations':2999,
           'actual_trees':2999,'feature_count':95,'categorical_feature_count':23,'native_categorical_feature_count':23,
           'sample_weight':WEIGHT_FORMULA,'weight_argument_passed_to_pool':True,'parameters':copy.deepcopy(params),
           'baseline_parameters':copy.deepcopy(params),'changed_parameter_names':[],'removed_feature_names':[],
           'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_boosting_type':'Plain',
           'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'b'*64,'target_formula':TARGET_FORMULA,'inverse_formula':INVERSE_FORMULA,
           'raw_z_diagnostics':{'synthetic':True},'runner_elapsed_seconds':1.0,'fit_rows':5,
           'fit_weight_diagnostics':copy.deepcopy(diag),'component_source_sha256':{'synthetic/helper.py':'f'*64},
           'geo_rule_target_weight_fields_are_provenance_only':True,**dict.fromkeys(AREA_HASH_FIELDS,'c'*64)}
    area={k:copy.deepcopy(model[k]) for k in (*AREA_HASH_FIELDS,'fit_weight_diagnostics','target_formula','inverse_formula','sample_weight')}
    area.update(raw_z_units=RAW_Z_UNITS,area_weight_factor_recomputed_from_original_X=True,
                Pool_weight_usage_verified_from_frozen_source_and_saved_metadata=True,historical_fit_not_reexecuted=True)
    state={k:copy.deepcopy(model[k]) for k in (*AREA_HASH_FIELDS,'fit_weight_diagnostics','target_formula','inverse_formula',
             'parameters','changed_parameter_names','removed_feature_names','target_mode')}
    state.update(source_sha256=copy.deepcopy(model['component_source_sha256']),all_feature_names=['x'+str(i) for i in range(95)],
                 categorical_columns=['x'+str(i) for i in range(23)],geo_state={'all_feature_names':['x'+str(i) for i in range(95)]})
    state['state_content_sha256']=hashlib.sha256(json.dumps(state,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
    model['feature_state_content_sha256']=state['state_content_sha256']
    proof={'replays':{'synthetic':True},'saved_raw_z_bitwise_equal':True,'removed_feature_names':[],'raw_z_units':RAW_Z_UNITS,
           'native_model_capacity':{k:model[k] for k in CAPACITY_FIELDS},'independent_area_weight_evidence':area,
           'independent_fit_state_evidence':{'fit_rows':5,'base_feature_count':95,'feature_count':95,'categorical_count':23,
               'removed_feature_names':[],'fit_state_reconstructed_independently':True,'fit_model_feature_values_sha256':'d'*64,
               'independent_area_weight_evidence':copy.deepcopy(area)}}
    check_capacity(model,proof);check_area_evidence(model,proof,state);checks.append('native6_and_area_state_model_proof_exact')
    for key,value in [('actual_boosting_type','Ordered'),('maximum_tree_leaf_count',257)]:
        bad=copy.deepcopy(model);bad[key]=value
        reject('wrong_native_'+key,lambda bad=bad:check_capacity(bad,proof))
    for key,value in [('fit_factor_sha256','d'*64),('sample_weight','uniform'),('native_categorical_feature_count',19)]:
        bad=copy.deepcopy(model);bad[key]=value
        reject('wrong_area_metadata_'+key,lambda bad=bad:check_area_evidence(bad,proof,state))
    bad=copy.deepcopy(state);bad['fit_sample_weights_sha256']='e'*64
    reject('tampered_json_feature_state',lambda:check_area_evidence(model,proof,bad))
    bad=copy.deepcopy(proof);bad['raw_z_units']='million_TWD'
    reject('wrong_raw_unit_scale',lambda:check_area_evidence(model,bad,state))
    initial=dict.fromkeys(RESEARCH_BINDING_FILES,'a'*64)
    record={'round':19,'status':'target_free_effective_area_support_audit','feature_values_modified':False,'prices_None_R1_transform_used':True,
       'read_scope':{'target_cells_decoded':0,'train_test_target_free_caches_only':True,'query_test_weights_descriptive_not_training':True,
          'MAE_computed':False,'candidate_fitted':False,'model_or_prediction_loaded':False,'raw_price_cache_or_Excel_loaded':False,'valid_dataset_loaded':False},
       'alignment':{'canonical_rows':455255,'fit_rows':240316,'query_rows':81234,'indices_and_dates_match_released_R6':True,
                    'original_saved_mapping_reused':True,'target_dedup_recomputed_here':False},
       'source_sha256':{RESEARCH_FILES[0]:'a'*64,**{f'synthetic/source{i}.py':'b'*64 for i in range(7)}},
       'evidence_sha256':{'synthetic/evidence.md':'c'*64},
       'inputs':{f'synthetic/input{i}.bin':{'path':f'synthetic/input{i}.bin','sha256':'d'*64,'bytes':8} for i in range(11)}}
    mapping=research_mapping(record,initial)
    check('research3_plus8_plus1_plus11_merges22_with_receipt',len(mapping)==22 and RESEARCH_FILES[2] in mapping)
    rp={'research_dependency_sha256':mapping,'dependency_sha256':mapping}
    validate_research_map(rp,mapping);checks.append('complete_research_subset')
    bad=copy.deepcopy(record);bad['read_scope']['target_cells_decoded']=1
    reject('research_target_decode_rejected',lambda:research_mapping(bad,initial))
    bad=copy.deepcopy(record);bad['inputs'].pop(next(iter(bad['inputs'])))
    reject('research_missing_input_rejected',lambda:research_mapping(bad,initial))
    bad=copy.deepcopy(record);bad['source_sha256'][RESEARCH_FILES[0]]='e'*64
    reject('research_conflicting_recipe_rejected',lambda:research_mapping(bad,initial))
    check('six_owned_sources_and_whole_research_copyable',len(SOURCES)==6 and all(copy_scope(n) for n in SOURCES+DOCS+RESEARCH_FILES))
    check('external_maps_caches_mutables_not_copyable',not any(copy_scope(n) for n in
          ('CURRENT_STATUS.md','runs/versions.json','data_cache/train_features.pkl','data_external/geography/taiwan-220101.osm.pbf',
           'outputs/round6/geography/final/manifest.json','src/archive_round18.py')))
    reject('traversal_path_refused',lambda:normalized('../bad'))
    bindings=Bindings();bindings.declare('runs/versions.json','a'*64)
    check('registry_context_not_live_binding',not bindings.values and 'runs/versions.json' in bindings.mutable_declarations)
    report={'round':19,'candidate':CANDIDATE,'accepted':False,'currently_recommended_submission':'outputs/submission_v3.csv',
            'stopping_reason':accepted['stopping_reason'],'deployment_decision':'keep_v3','new_submission_created':False,
            'historically_untouched':False,'public_score_used':False,'private_score_known':False,
            'report_only_reads_saved_JSON_and_hashes_binary_inputs':True,'stages':[]}
    models={stage:copy.deepcopy(model) for stage in STAGES}
    proofs={stage:copy.deepcopy(proof) for stage in STAGES}
    proof_shas=dict.fromkeys(STAGES,'a'*64)
    for stage in STAGES:
        row={'stage':stage,'status':accepted['stage_status'][stage],'dates':plan['splits'][stage],'fit_rows':5,'query_rows':3,
             'metrics':copy.deepcopy(results[stage]['metrics']),'decision':copy.deepcopy(results[stage]['decision']),
             'model':{k:model[k] for k in MODEL_SUMMARY_FIELDS},
             'independent_review':{'proof_path':'outputs/round19/audit/'+stage+'_replay.json','proof_sha256':'a'*64,
               'replays':proof['replays'],'raw_z_bitwise_equal':True,'native_model_capacity':proof['native_model_capacity'],
               'removed_feature_names':[],'independent_fit_state_evidence':proof['independent_fit_state_evidence'],
               'independent_area_weight_evidence':area}}
        report['stages'].append(row)
    check_report(report,plan,accepted,results,models,proofs,proof_shas);checks.append('all_three_report_metrics_models_and_unit_proofs_exact')
    bad=copy.deepcopy(report);del bad['stages'][1]['independent_review']['independent_area_weight_evidence']
    reject('report_omits_later_independent_unit_evidence',lambda:check_report(bad,plan,accepted,results,models,proofs,proof_shas))
    bad=copy.deepcopy(report);bad['stages'][2]['metrics']=None
    reject('report_hides_completed_auxiliary',lambda:check_report(bad,plan,accepted,results,models,proofs,proof_shas))
    bad=copy.deepcopy(report);bad['stages'][1]['metrics']=copy.deepcopy(report['stages'][0]['metrics'])
    reject('report_mixes_stage_metrics',lambda:check_report(bad,plan,accepted,results,models,proofs,proof_shas))
    return {'synthetic_only':True,'complete_bounded_self_check_executed':True,
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
