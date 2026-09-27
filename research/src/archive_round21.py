"""Archive the finalized R21 development rejection; stdlib and byte hashes only.
Explicit --create / --verify / --self-check. No model or array deserialization,
training, scoring, release, registry write, cleanup, or old-source imports.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round21'
DEST = ROOT / 'runs/round21_experiments'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, CANDIDATE = 'v3_control', 'fine_grid_geography_blend'
SOURCES = ('src/round21_fine_grid_geo.py', 'src/round21_experiment.py',
           'src/round21_verify_backtest.py', 'src/round21_report.py',
           'src/archive_round21.py', 'src/round21_grid_audit.py')
DOCS = ('reports/round21_plan.md', 'reports/round21_method_review.md',
        'reports/round21_pipeline_review.md', 'reports/round21_work_log.md',
        'reports/round21_report.md', 'reports/round21_results.json', 'reports/round21_results.sha256')
REMOVED_FEATURES = []
ADDED_FEATURES = ['geo_grid_0p005', 'geo_grid_0p005_shift']
CHANGED_PARAMETERS = []
LR_CONTRACT = {'expected': .065, 'rtol': 1e-7, 'atol': 0.0}
INVENTORY_PATH = 'outputs/round21_research/grid_audit.json'
RESEARCH_FILES = ('outputs/round21_research/grid_audit.json', 'outputs/round21_research/grid_audit.sha256',
    'outputs/round21_research/self_check.json', 'outputs/round21_research/self_check.sha256',
    'outputs/round21_research/method_before_audit.md', 'outputs/round21_research/logs/self_check.log',
    'outputs/round21_research/logs/audit.log')
MODEL_FILES = {'model.cbm', 'feature_state.json', 'predictions.npy', 'raw_z.npy',
               'fit_indices.npy', 'query_indices.npy', 'metadata.json', 'metadata.sha256'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                  'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
                  'actual_learning_rate')
MODEL_SUMMARY_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names', 'added_feature_names',
    'baseline_feature_count', 'original95_preserved',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fitted_actual_learning_rate', 'native_categorical_feature_count',
    'native_learning_rate_comparison_rtol', 'native_learning_rate_comparison_atol',
    'geo_encoder_rule_training_fields_are_baseline_provenance_only') + CAPACITY_FIELDS
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = '# R21 development-rejection snapshot\n\nThis is an experiment archive, not a release. The completed development\ncandidate failed the fixed gates. Confirmation/auxiliary price fits/evaluations\nand final fitting were not run; outputs/submission_v3.csv remains recommended.\nNo new submission or registry update is produced by this utility.\n\nThe component preserves the original R6 95 features and adds two deterministic\n0.005-degree categorical grids (regular and both coordinates shifted +0.0025).\nThe stored float32 coordinates are cast to float64 for the two new formulas;\noriginal geo_grid_002 stays unchanged. Missing positions receive missing grids.\nThere is no support threshold, fitted vocabulary filter, or coordinate repair.\nThe resulting schema is 97 features / 25 categories. All R6 passed parameters\nremain unchanged: raw y_TWD/1e6, uniform weights, 2999 trees, learning rate .065,\ndepth 8, Plain and SymmetricTree. Original X87 reference/scoring remains intact.\nThe fitted and saved native learning rates use the fixed 1e-7 relative tolerance.\n\nSix owned sources, documents, the entire round21 and round21_research output\nfolders (including pipeline_runner.py and logs) are copied. Research provenance\nincludes the declared 29 identities and all nested audit input/evidence hashes.\nThe orchestrator is preserved as bytes, not imported as model source closure.\nThere was no pre-score failure or retry contract in R21. Earlier diagnostics\nremain historical bytes, never replacement declarations of current sources.\nOriginal data, old models, all four annual geographic chains and other declared\ndependencies are external SHA/size guards; this archive is not self-contained.\n\nCreate requires one exact CLOSED work-log line in the same bytes archived.\nVerification reads JSON and hashes files; it never deserializes arrays/models,\nrefits models or recomputes errors. Frozen copied originals are also checked.\nversions.json is creation-time context only; it and CURRENT_STATUS.md are not\npermanent guards. Snapshots and partial staging are never overwritten/deleted.\n'



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
    return (name in SOURCES or name.startswith('outputs/round21/')
            or name.startswith('outputs/round21_research/')
            or name.startswith('reports/round21_') or name == 'README_round21.md')


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


def validate_current_stage_names(names):
    need(set(names)=={'development'}, 'Current staging or unauthorized phase must be preserved and inspected')


def validate_current_paths(names):
    for name in names:
        normalized(name)
        need(not any(part.startswith(('.', '_stage_')) for part in name.split('/'))
             and not name.lower().endswith(('.tmp','.part','.lock')),
             'Current incomplete staging must be preserved and inspected: '+name)


def inventory():
    need(OUT.is_dir(), 'No R21 outputs')
    for area in ('models', 'evaluation'):
        folder=OUT/area
        need(folder.is_dir(), 'Missing current '+area)
        validate_current_stage_names({p.name for p in folder.iterdir()})
    for name in ('final','release'):
        need(not (OUT/name).exists(), 'R21 rejection cannot contain '+name)
    need(not list((ROOT/'reports').glob('.round21-report-*')), 'Partial report staging requires inspection; keep it')
    model=OUT/'models/development'; evaluation=OUT/'evaluation/development'
    need({p.name for p in model.iterdir()} == MODEL_FILES and all(p.is_file() for p in model.iterdir()), 'Incomplete model inventory')
    need({p.name for p in evaluation.iterdir()} == {CONTROL+'.npy',CANDIDATE+'.npy','results.json','results.sha256'}
         and all(p.is_file() for p in evaluation.iterdir()), 'Incomplete evaluation inventory')
    research = ROOT/'outputs/round21_research'
    need(research.is_dir(), 'Missing owned pure-X research')
    validate_current_paths({p.relative_to(OUT).as_posix() for p in OUT.rglob('*')})
    validate_current_paths({p.relative_to(research).as_posix() for p in research.rglob('*')})
    paths=set(OUT.rglob('*')) | set(research.rglob('*'))
    for path in (ROOT/'reports').glob('round21_*'):
        paths.add(path)
        if path.is_dir(): paths.update(path.rglob('*'))
    paths.update(ROOT/name for name in SOURCES)
    if (ROOT/'README_round21.md').exists(): paths.add(ROOT/'README_round21.md')
    files=set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink is not an archive artifact: '+str(path))
        if path.is_file():
            name=rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R21 copy scope')
            if name.startswith('reports/'):
                need(not name.lower().endswith(('.tmp','.part','.lock')), 'Partial report file')
            need(not (path.name.lower().startswith('submission') and path.suffix.lower()=='.csv'), 'No new submission in experiment archive')
            files.add(name)
    need(set(SOURCES).union(DOCS,RESEARCH_FILES) <= files, 'Required R21 source/document/research missing')
    return files


def validate_development_decision(decision):
    keys={'overall_improves_at_least_1pct','weighted_improves_at_least_1pct',
          'quarters_deteriorate_at_most_5pct','large_strata_deteriorate_at_most_5pct'}
    need(set(decision['checks'])==keys and all(type(v) is bool for v in decision['checks'].values()),
         'Fixed development gate inventory differs')
    passed=all(decision['checks'].values())
    need(decision['passed'] is passed and decision['selected_candidate']==(CANDIDATE if passed else CONTROL),
         'Saved gate booleans and selected candidate disagree')


def validate_parameter_change(baseline, params):
    need(params == baseline and baseline['iterations']==2999 and baseline['learning_rate']==.065
         and baseline['depth']==8 and baseline['boosting_type']=='Plain',
         'All original R6 passed parameters must remain unchanged')


def validate_grid_rules(rules):
    need(rules['feature_count']==97 and rules['categorical_feature_count']==25
         and rules['baseline_feature_count']==95 and rules['original95_preserved'] is True
         and rules['existing_95_values_dtypes_order_and_index_unchanged'] is True
         and rules['removed_feature_names']==[] and rules['added_feature_names']==ADDED_FEATURES,
         'The exact original95 plus two-grid schema is required')
    need(rules['grid_degrees']==.005 and rules['grid_shift_degrees']==.0025
         and rules['coordinate_input']=='existing float32 geography coordinates cast to float64'
         and rules['regular_grid_formula']=='floor(lat_float64 / 0.005):floor(lon_float64 / 0.005)'
         and rules['shifted_grid_formula']=='floor((lat_float64 + 0.0025) / 0.005):floor((lon_float64 + 0.0025) / 0.005)'
         and rules['missing_grid']=='__MISSING__'
         and rules['unknown_query_grid_policy']=='keep computed string; no fit-vocabulary filtering'
         and rules['original_coarse_grid_policy']=='preserve existing geo_grid_002; never rebuild from float32'
         and rules['grid_rounding_epsilon_or_boundary_adjustment'] is False
         and rules['fit_grid_counts_or_vocabulary_filter'] is False, 'Fixed grid rules differ')


def rejection_contract(plan, frozen, accepted, result):
    need(plan['round'] == frozen['round'] == accepted['round'] == result['round'] == 21,
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
    need(plan['feature_count']==97 and plan['categorical_count']==25
         and plan['parameters']['random_seed']==42 and plan['parameters']['thread_count']==4
         and plan['parameters']['loss_function']=='MAE'
         and plan['component']=='geography_fine_grid_catboost'
         and plan['sample_weight']=='uniform' and plan['target_mode']=='raw', 'Fixed raw model contract differs')
    validate_parameter_change(plan['baseline_parameters'],plan['parameters'])
    need(plan['changed_parameter_names']==CHANGED_PARAMETERS
         and plan['model_hyperparameter_change']=={}
         and plan['capacity_constraint']=={'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'}
         and plan['native_learning_rate_contract']==LR_CONTRACT, 'Fixed parameter/capacity contract differs')
    need(plan['removed_feature_names']==[] and plan['feature_rules']['removed_feature_names']==[], 'Original 95 features must remain')
    need(plan['added_feature_names']==ADDED_FEATURES, 'Two fixed grids required')
    validate_grid_rules(plan['feature_rules'])
    validate_development_decision(result['decision'])
    need(plan['target_formula'] == 'y_TWD / 1e6'
         and plan['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong target/inverse')
    need(result['fit_rows'] == plan['split_contract']['development']['fit_rows']
         and result['query_rows'] == plan['split_contract']['development']['query_rows'], 'Rows differ')
    for field in ('formulas', 'thresholds'):
        need(result[field] == frozen[field] == plan[field], 'Frozen protocol differs: ' + field)
    need(frozen['parameters'] == plan['parameters'] and result['target_group_weights']
         == frozen['target_group_weights'] == plan['target_group_weights'], 'Frozen parameters/weights differ')


def validate_learning_rates(saved,fitted):
    for rate in (saved,fitted):
        need(isinstance(rate,(int,float)) and not isinstance(rate,bool) and math.isfinite(rate)
             and abs(rate-.065)<=1e-7*abs(.065), 'Native learning rate violates fixed tolerance')
    need(abs(fitted-saved)<=1e-7*abs(saved), 'Fitted/saved learning rates exceed the fixed reload tolerance')


def check_capacity(model,proof):
    capacity={key:model[key] for key in CAPACITY_FIELDS}
    need(capacity['actual_model_depth']==8 and capacity['actual_boosting_type']=='Plain'
         and capacity['actual_grow_policy']=='SymmetricTree' and capacity['actual_trees']==2999,
         'Wrong native CatBoost capacity')
    need(type(capacity['maximum_tree_leaf_count']) is int and 1<=capacity['maximum_tree_leaf_count']<=256,
         'Wrong native leaf-cap evidence')
    need(isinstance(capacity['tree_leaf_counts_sha256'],str)
         and re.fullmatch('[0-9a-f]{64}',capacity['tree_leaf_counts_sha256']), 'Invalid LE-int64 leaf digest')
    validate_learning_rates(capacity['actual_learning_rate'],model['fitted_actual_learning_rate'])
    need(proof['native_model_capacity']==capacity, 'Proof/model saved native capacity differs')
    for value in (model,proof):
        need(value['native_categorical_feature_count']==25
             and value['native_learning_rate_comparison_rtol']==1e-7
             and value['native_learning_rate_comparison_atol']==0.0, 'Native category/LR contract differs')


def check_report(report, plan, accepted, result, model, proof, proof_sha):
    need(report['round'] == 21 and report['accepted'] is False
         and report['candidate'] == CANDIDATE
         and report['currently_recommended_submission'] == 'outputs/submission_v3.csv'
         and report['stopping_reason'] == accepted['stopping_reason']
         and report['deployment_decision'] == 'keep_v3', 'Report decision differs')
    for key in ('new_submission_created', 'historically_untouched', 'public_score_used', 'private_score_known'):
        need(report[key] is False, 'Report policy differs: ' + key)
    need('pre_score_retry_record' not in report, 'R21 must not inherit an R20 retry record')
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
                'proof_path': 'outputs/round21/audit/development_replay.json', 'proof_sha256': proof_sha,
                'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal'],
                'native_model_capacity':proof['native_model_capacity'],
                'removed_feature_names':proof['removed_feature_names'],
                'added_feature_names':proof['added_feature_names'],
                'independent_fit_state_evidence':proof['independent_fit_state_evidence']},
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


def research_expectations(source_text):
    """Read frozen literal declarations only, never import the ML driver."""
    found={}
    for node in ast.parse(source_text).body:
        if isinstance(node,ast.Assign):
            for target in node.targets:
                if isinstance(target,ast.Name) and target.id in ('RESEARCH_FILES','RESEARCH_EXPECTED_SHA'):
                    need(target.id not in found,'Duplicate research declaration')
                    found[target.id]=ast.literal_eval(node.value)
    need(set(found)=={'RESEARCH_FILES','RESEARCH_EXPECTED_SHA'},'Missing literal R21 research declarations')
    seeds,expected=found['RESEARCH_FILES'],found['RESEARCH_EXPECTED_SHA']
    need(isinstance(seeds,(tuple,list)) and len(seeds)>0 and len(set(seeds))==len(seeds)
         and isinstance(expected,dict) and set(seeds)==set(expected), 'Research seed and expected map inventories differ')
    return {name:expected[name] for name in seeds}


def validate_research(research,dependencies,declared,inventory):
    """Compare frozen pure-X provenance; never recompute inventory or price scores."""
    need(isinstance(research,dict) and research and research==declared, 'Frozen research differs from literal driver map')
    seeds=list(declared)
    need(INVENTORY_PATH in seeds and INVENTORY_PATH.replace('.json','.sha256') in seeds,
         'Inventory JSON and receipt must be frozen research seeds')
    need(all(dependencies.get(k)==v for k,v in research.items()), 'Research outside frozen dependencies')
    expected={name:research[name] for name in seeds}
    def add(name,digest):
        need(isinstance(name,str) and isinstance(digest,str) and len(digest)==64
             and all(c in '0123456789abcdef' for c in digest), 'Invalid research identity')
        need(name not in expected or expected[name]==digest, 'Conflicting pure-X provenance')
        expected[name]=digest
    for name,digest in expected.copy().items(): add(name,digest)
    need(inventory['round']==21 and inventory['status']=='completed_pure_X_only'
         and isinstance(inventory['scope'],dict) and isinstance(inventory['rules'],dict), 'Wrong pure-X inventory contract')
    for field in ('source_sha256','evidence_sha256'):
        for name,digest in inventory[field].items(): add(name,digest)
    for item in inventory['inputs'].values():
        need(type(item['bytes']) is int and item['bytes']>=0, 'Invalid research input size')
        add(item['path'],item['sha256'])
    synthetic=inventory['synthetic_evidence']
    add(synthetic['path'],synthetic['sha256'])
    need(synthetic['path'].endswith('.json') and synthetic['path'][:-5]+'.sha256' in research,
         'Pure-X synthetic evidence receipt omitted')
    need(research==expected, 'Frozen research closure differs from seeds and inventory evidence')
    return {'frozen_research_dependency_count':len(research),'dependency_sha256':copy.deepcopy(research),
      'required_research_seed_paths':list(seeds),'research_type':'pure_X_geographic_grid_support_inventory',
      'inventory_path':INVENTORY_PATH,'inventory_sha256':research[INVENTORY_PATH],
      'inventory_status':inventory['status'],'inventory_scope':copy.deepcopy(inventory['scope']),
      'inventory_alignment':copy.deepcopy(inventory['alignment']),'inventory_rules':copy.deepcopy(inventory['rules']),
      'inventory_limitations':copy.deepcopy(inventory['limitations']),
      'inventory_synthetic_evidence':copy.deepcopy(inventory['synthetic_evidence']),
      'report_recomputed_inventory_or_price_metrics':False}


def check_research(plan, bindings):
    name = 'src/round21_experiment.py'
    bindings.add(name, plan['source_sha256'][name])
    declared = research_expectations(safe(name).read_text(encoding='utf-8-sig'))
    need(len(declared)==29, 'Frozen R21 research inventory must contain 29 identities')
    evidence = bindings.authority(INVENTORY_PATH)
    summary = validate_research(plan['research_dependency_sha256'], plan['dependency_sha256'], declared, evidence)
    bindings.mapping(declared)
    for item in evidence['inputs'].values():
        need(bindings.values[item['path']]['bytes']==item['bytes'], 'Research input byte length differs')
    for field in ('source_sha256', 'evidence_sha256'):
        bindings.mapping(evidence[field])
    synthetic=evidence['synthetic_evidence']
    bindings.add(synthetic['path'],synthetic['sha256'])
    bindings.authority(synthetic['path'])
    return summary


def collect():
    """Read finalized authorities and hashes. Never inspect npy/model contents."""
    files = inventory()
    b = Bindings()
    work_log_name = 'reports/round21_work_log.md'
    work_log_bytes = safe(work_log_name).read_bytes()
    b.add(work_log_name, closed_work_log_sha256(work_log_bytes))
    authority_paths = {
        'plan': 'outputs/round21/candidate_plan.json',
        'frozen': 'outputs/round21/frozen_candidate.json',
        'accepted': 'outputs/round21/accepted_candidate.json',
        'result': 'outputs/round21/evaluation/development/results.json',
        'model': 'outputs/round21/models/development/metadata.json',
        'proof': 'outputs/round21/audit/development_replay.json',
        'report': 'reports/round21_results.json',
    }
    values = {key: b.authority(name) for key, name in authority_paths.items()}
    plan, frozen, accepted, result, model, proof, report = (values[k] for k in authority_paths)
    digests = {key: b.values[name]['sha256'] for key, name in authority_paths.items()}
    rejection_contract(plan, frozen, accepted, result)
    for value in (frozen, accepted, result, model, proof):
        need(value['plan_sha256'] == digests['plan'], 'Authority refers to another R21 plan')
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
    research_summary = check_research(plan, b)
    need(report['research_summary'] == research_summary, 'Report research scope differs')
    b.add('src/round21_report.py', report['source_input_sha256']['src/round21_report.py'])
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
    expected_outputs = {'outputs/round21/evaluation/development/' + name + '.npy'
                        for name in (CONTROL, CANDIDATE)}
    need(set(result['output_artifact_sha256']) == expected_outputs, 'Unexpected prediction inventory')
    need(model['round'] == 21 and model['stage'] == 'development'
         and model['component'] == plan['component'] and model['parameters'] == plan['parameters'],
         'Model configuration differs')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 2999
         and model['family'] == 'catboost'
         and model['weight_argument_passed_to_pool'] is False
         and model['feature_rules'] == plan['feature_rules']
         and model['feature_count'] == 97 and model['categorical_feature_count'] == 25
         and model['sample_weight'] == 'uniform' and model['target_mode'] == plan['target_mode']
         and model['target_formula'] == plan['target_formula']
         and model['inverse_formula'] == plan['inverse_formula'], 'Model scale/schema differs')
    need(model['baseline_parameters'] == proof['baseline_parameters'] == plan['baseline_parameters']
         and model['changed_parameter_names'] == proof['changed_parameter_names']
         == plan['changed_parameter_names'] == CHANGED_PARAMETERS
         and model['removed_feature_names'] == proof['removed_feature_names'] == plan['removed_feature_names'] == REMOVED_FEATURES,
         'Fixed parameter or original-feature evidence differs')
    need(model['added_feature_names']==proof['added_feature_names']==plan['added_feature_names']==ADDED_FEATURES
         and model['baseline_feature_count']==95 and model['original95_preserved'] is True, 'Added-grid preservation differs')
    validate_parameter_change(model['baseline_parameters'],model['parameters'])
    check_capacity(model, proof)
    need(model['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and model['feature_rules']['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and model['feature_rules']['iterations']==2999
         and model['feature_rules']['learning_rate']==.065
         and model['feature_rules']['geo_encoder_rules']['iterations']==2999,
         'Original R6 encoder provenance must not replace current training settings')
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
    need(proof['round'] == 21 and proof['stage'] == 'development' and proof['status'] == 'passed'
         and proof['candidate_accepted_in_this_stage'] is False and proof['saved_raw_z_bitwise_equal'] is True,
         'Independent proof does not certify this rejection')
    need(proof['independent_metrics'] == result['metrics']
         and proof['independent_decision'] == result['decision']
         and proof['raw_z_diagnostics'] == model['raw_z_diagnostics'], 'Proof/report payload mismatch')
    for key in ('fit_performed', 'new_window_scored', 'public_score_used', 'historically_untouched'):
        need(proof[key] is False, 'Unexpected independent review behavior: ' + key)
    need(proof['model_family'] == 'catboost' and proof['model_format'] == 'native_CatBoost_cbm'
         and proof['raw_z_units'] == 'million_TWD' and proof['actual_iterations'] == 2999
         and proof['feature_count'] == 97 and proof['categorical_count'] == 25,
         'Independent raw geographic CatBoost schema differs')
    for key in ('fit_feature_state_reconstructed_independently',
                'query_model_view_checked_independently',
                'original_95_values_dtypes_order_index_preserved',
                'original_87_reference_and_scoring_keys_unchanged'):
        need(proof[key] is True, 'Independent feature proof differs: ' + key)
    state_evidence = proof['independent_fit_state_evidence']
    need(state_evidence['fit_rows'] == spec['fit_rows']
         and state_evidence['base_feature_count'] == 95
         and state_evidence['feature_count'] == 97 and state_evidence['categorical_count'] == 25
         and state_evidence['removed_feature_names'] == REMOVED_FEATURES
         and state_evidence['fit_state_reconstructed_independently'] is True
         and state_evidence['fit_model_feature_values_sha256'] == model['fit_model_feature_values_sha256'],
         'Independent fit-only geographic state differs')
    need(state_evidence['added_feature_names']==ADDED_FEATURES
         and state_evidence['grids_reconstructed_independently_from_stored_float32'] is True
         and re.fullmatch('[0-9a-f]{64}',state_evidence['fit_added_grid_values_sha256']),
         'Independent reconstruction of the added grids is missing')
    b.add('src/round21_verify_backtest.py', proof['source_sha256'])
    check_report(report, plan, accepted, result, model, proof, digests['proof'])
    # These three source maps are authoritative; verify that the independent
    # replay actually bound the artifacts it says it replayed, without loading them.
    required_proof = {authority_paths[k] for k in ('plan', 'frozen', 'accepted', 'result', 'model')}
    required_proof.update('outputs/round21/models/development/' + name
                          for name in MODEL_FILES)
    required_proof.update(expected_outputs)
    need(required_proof <= set(proof['source_input_sha256']), 'Replay input closure is incomplete')
    required_report = set(required_proof) | {authority_paths['accepted'], authority_paths['proof'],
                                           'src/round21_report.py'}
    need(required_report <= set(report['source_input_sha256']), 'Report input closure is incomplete')
    # Attempts are copied, not interpreted as current source bindings.
    for name in sorted(files):
        b.add(name)
        if name.endswith('.sha256'):
            path = safe(name)
            target = path.with_suffix('.json')
            need(target.is_file(), 'Historical/current receipt has no JSON: ' + name)
            locked(target)  # Byte receipt only; do not traverse historical source maps.
    need(inventory() == files, 'R21 inventory changed during collection')
    need(all(not copy_scope(name) or name in files for name in b.values), 'Owned dependency omitted from copy inventory')
    copied = {name: b.values[name].copy() for name in sorted(files)}
    external = {name: item.copy() for name, item in sorted(b.values.items()) if name not in copied}
    need(not MUTABLE.intersection(set(copied) | set(external)), 'Mutable entry point became permanent')
    registry = safe('runs/versions.json')
    context = {'path': 'runs/versions.json', 'sha256': sha(registry), 'bytes': registry.stat().st_size,
               'plan_time_sha256': plan['registry_sha256_at_plan'],
               'authority_observed_sha256': sorted(b.mutable_declarations.get('runs/versions.json', set())),
               'verification_policy': 'creation-time observation only; never a future external guard'}
    state = {'round': 21, 'is_release': False, 'accepted_backtest': False,
             'candidate': CANDIDATE, 'selected_candidate': CONTROL,
             'stopping_reason': accepted['stopping_reason'], 'stage_status': accepted['stage_status'],
             'recommended_version': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
             'new_submission_created': False, 'registry_modified': False,
             'plan_sha256': digests['plan'], 'frozen_candidate_sha256': digests['frozen'],
             'accepted_candidate_sha256': digests['accepted'], 'development_result_sha256': digests['result'],
             'independent_replay_sha256': digests['proof'], 'report_json_sha256': digests['report'],
             'source_path': 'src/archive_round21.py', 'source_sha256': sha(__file__),
             'property_labels_or_prediction_values_read_by_archiver': False,
             'array_or_model_deserialization_performed': False, 'models_refitted_or_MAE_recomputed': False,
             'index_identity_uses_saved_contract_and_replay_not_array_deserialization': True,
             'historically_untouched': False, 'historical_attempts_kept_without_current_source_enforcement': True,
             'registry_context': context, 'current_status_is_not_bound': True,
             'research_dependency_sha256': plan['research_dependency_sha256'],
             'research_is_target_free_descriptive_evidence': True,
             'new_research_folder_and_receipts_preserved': True, 'research_summary': research_summary}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round21_development_rejection_experiments', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 21 and state['is_release'] is False and state['accepted_backtest'] is False
         and state['selected_candidate'] == CONTROL and state['stopping_reason'] == 'development_price_gates_failed'
         and state['recommended_version'] == 'v3' and state['new_submission_created'] is False,
         'Snapshot does not describe the finalized development rejection')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copied and external paths overlap')
    names = set(copied) | set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names} & {n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= set(copied) and all(copy_scope(n) for n in copied),
         'Wrong R21 copy inventory')
    need(all(not copy_scope(n) for n in external), 'Owned R21 artifact left external')
    need(manifest['copied_file_count'] == len(copied)
         and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(item['bytes'] for item in copied.values()),
         'Manifest count/size mismatch')
    for name, item in {**copied, **external}.items():
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid file identity: ' + name)
    for key, name in [('plan_sha256', 'outputs/round21/candidate_plan.json'),
                      ('frozen_candidate_sha256', 'outputs/round21/frozen_candidate.json'),
                      ('accepted_candidate_sha256', 'outputs/round21/accepted_candidate.json'),
                      ('development_result_sha256', 'outputs/round21/evaluation/development/results.json'),
                      ('independent_replay_sha256', 'outputs/round21/audit/development_replay.json'),
                      ('report_json_sha256', 'reports/round21_results.json'),
                      ('source_sha256', 'src/archive_round21.py')]:
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
        need(inventory() == set(copied), 'Original R21 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}


def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round21_experiments_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round21_experiments_staging_' + uuid.uuid4().hex)
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
    manifest = {'kind': 'round21_development_rejection_experiments',
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
    checks=work_log_self_check()
    def check(name,value):
        need(bool(value),'Synthetic check failed: '+name); checks.append(name)
    def reject(name,fn):
        try: fn()
        except (RuntimeError,KeyError,TypeError,ValueError): checks.append(name)
        else: raise AssertionError('Expected rejection: '+name)
    baseline={'iterations':2999,'learning_rate':.065,'depth':8,'boosting_type':'Plain'}
    validate_parameter_change(baseline,dict(baseline)); checks.append('all_R6_parameters_preserved')
    for key,value in [('iterations',5998),('learning_rate',.0325),('depth',6),('boosting_type','Ordered')]:
        wrong={**baseline,key:value}
        reject('changed_'+key+'_rejected',lambda wrong=wrong:validate_parameter_change(baseline,wrong))
    model={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
      'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain',
      'actual_learning_rate':.06499999762,'fitted_actual_learning_rate':.06499999761581421,
      'native_categorical_feature_count':25,'native_learning_rate_comparison_rtol':1e-7,
      'native_learning_rate_comparison_atol':0.}
    proof={k:model[k] for k in ('native_categorical_feature_count','native_learning_rate_comparison_rtol','native_learning_rate_comparison_atol')}
    proof['native_model_capacity']={k:model[k] for k in CAPACITY_FIELDS}
    check_capacity(model,proof);checks.append('native7_2999_rate065_and_25_categories')
    for key,value in [('actual_trees',5998),('native_categorical_feature_count',23),('actual_learning_rate',.0325),('maximum_tree_leaf_count',257)]:
        wrong={**model,key:value}
        reject('wrong_native_'+key,lambda wrong=wrong:check_capacity(wrong,proof))
    rules={'feature_count':97,'categorical_feature_count':25,'baseline_feature_count':95,'original95_preserved':True,
      'existing_95_values_dtypes_order_and_index_unchanged':True,'removed_feature_names':[], 'added_feature_names':list(ADDED_FEATURES),
      'grid_degrees':.005,'grid_shift_degrees':.0025,'coordinate_input':'existing float32 geography coordinates cast to float64',
      'regular_grid_formula':'floor(lat_float64 / 0.005):floor(lon_float64 / 0.005)',
      'shifted_grid_formula':'floor((lat_float64 + 0.0025) / 0.005):floor((lon_float64 + 0.0025) / 0.005)',
      'missing_grid':'__MISSING__','unknown_query_grid_policy':'keep computed string; no fit-vocabulary filtering',
      'original_coarse_grid_policy':'preserve existing geo_grid_002; never rebuild from float32',
      'grid_rounding_epsilon_or_boundary_adjustment':False,'fit_grid_counts_or_vocabulary_filter':False}
    validate_grid_rules(rules);checks.append('fixed_both_plus_grids_and_original95')
    for key,value in [('grid_shift_degrees',-.0025),('feature_count',95),('fit_grid_counts_or_vocabulary_filter',True)]:
        wrong={**rules,key:value};reject('wrong_grid_'+key,lambda wrong=wrong:validate_grid_rules(wrong))
    mapping={INVENTORY_PATH:'a'*64,INVENTORY_PATH.replace('.json','.sha256'):'b'*64,
      'outputs/round21_research/self_check.json':'c'*64,'outputs/round21_research/self_check.sha256':'d'*64,
      'old/geo.pkl':'e'*64,'src/round21_grid_audit.py':'f'*64}
    source='RESEARCH_FILES = '+repr(tuple(mapping))+'\nRESEARCH_EXPECTED_SHA = '+repr(mapping)
    check('literal_research_names_and_SHA_map',research_expectations(source)==mapping)
    inv={'round':21,'status':'completed_pure_X_only','scope':{},'rules':rules,'alignment':{},'limitations':[],
      'source_sha256':{'src/round21_grid_audit.py':'f'*64},'evidence_sha256':{},
      'inputs':{'geo':{'path':'old/geo.pkl','sha256':'e'*64,'bytes':10}},
      'synthetic_evidence':{'path':'outputs/round21_research/self_check.json','sha256':'c'*64}}
    summary=validate_research(mapping,mapping,mapping,inv)
    check('nested_research_and_synthetic_receipt_closure',summary['dependency_sha256']==mapping)
    wrong=copy.deepcopy(inv);wrong['inputs']['geo']['sha256']='0'*64
    reject('research_input_conflict_rejected',lambda:validate_research(mapping,mapping,mapping,wrong))
    reject('research_executable_expression_rejected',lambda:research_expectations('RESEARCH_FILES = dangerous()'))
    validate_current_stage_names({'development'});checks.append('development_only_inventory')
    reject('later_stage_rejected',lambda:validate_current_stage_names({'development','confirmation'}))
    reject('partial_staging_rejected',lambda:validate_current_paths({'models/.development-any/model.cbm'}))
    check('six_owned_sources_and_research_copy_scope',len(SOURCES)==6 and all(copy_scope(n) for n in SOURCES+DOCS+RESEARCH_FILES))
    check('runner_and_logs_preserved',copy_scope('outputs/round21/pipeline_runner.py') and copy_scope('outputs/round21/logs/report.log'))
    check('old_geo_and_mutable_not_copied',not any(copy_scope(n) for n in ('runs/versions.json','CURRENT_STATUS.md','src/round6_geo.py','outputs/round6/geography/final/train.pkl','outputs/round20_pre_score_failure_01/a.json')))
    binding=Bindings();binding.declare('src/a.py','a'*64);binding.values['src/a.py']={'sha256':'a'*64,'bytes':0}
    reject('conflicting_authority_rejected',lambda:binding.declare('src/a.py','b'*64))
    check('registry_is_context_only',binding.declare('runs/versions.json','c'*64) is False and 'runs/versions.json' not in binding.values)
    return {'synthetic_only':True,'real_authorities_labels_models_or_results_read':False,
      'files_written':0,'checks':checks,'count':len(checks),'script_sha256':sha(__file__)}


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

