"""Archive the finalized R20 development rejection; stdlib and byte hashes only.
Explicit --create / --verify / --self-check. No model or array deserialization,
training, scoring, release, registry write, cleanup, or old-source imports.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round20'
DEST = ROOT / 'runs/round20_experiments'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, CANDIDATE = 'v3_control', 'slow_catboost_geography_blend'
SOURCES = ('src/round20_slow_geo.py', 'src/round20_experiment.py',
           'src/round20_verify_backtest.py', 'src/round20_report.py', 'src/archive_round20.py')
DOCS = ('reports/round20_plan.md', 'reports/round20_method_review.md',
        'reports/round20_pipeline_review.md', 'reports/round20_work_log.md',
        'reports/round20_report.md', 'reports/round20_results.json', 'reports/round20_results.sha256')
REMOVED_FEATURES = []
CHANGED_PARAMETERS = ['iterations', 'learning_rate']
LR_CONTRACT = {'expected': .0325, 'rtol': 1e-7, 'atol': 0.0}
REPORT_SOURCE_SHA = 'a68e309312e951583917b654efe938cbe4c415c6ffc4023eeef39a0eace8cf0e'
MODEL_FILES = {'model.cbm', 'feature_state.json', 'predictions.npy', 'raw_z.npy',
               'fit_indices.npy', 'query_indices.npy', 'metadata.json', 'metadata.sha256'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                  'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
                  'actual_learning_rate')
MODEL_SUMMARY_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fitted_actual_learning_rate', 'native_categorical_feature_count',
    'native_learning_rate_comparison_rtol', 'native_learning_rate_comparison_atol',
    'geo_encoder_rule_training_fields_are_baseline_provenance_only') + CAPACITY_FIELDS
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
FAILURE_PREFIX = 'outputs/round20_pre_score_failure_01/'
FAILURE_RECORD_SHA = '5b7904b9d8b9d6199fec7f8d76ae137e4d8047f90b0f4d966d2536ebb30acf79'
FAILURE_STAGE = 'models/.development-2f476485b03540b6bec1cbb42b352d47/'
FAILURE_OUTPUT_NAMES = frozenset(('audit/preflight.json', 'audit/preflight.sha256',
    'candidate_plan.json', 'candidate_plan.sha256', 'logs/development_fit.log',
    'logs/plan.log', 'logs/self_check.log', FAILURE_STAGE+'feature_state.json', FAILURE_STAGE+'model.cbm'))
FAILURE_ORIGINAL_FILES = {
 'original_files/reports/round20_method_review.md': 'ab8d3762db687a7f6f7e7d2f8949e2b24f1384aac95aa5a7a5b95360402f795d',
 'original_files/reports/round20_pipeline_review.md': 'fc1c83b1157fc34f73ea6babe55993797147f1046dad6a3df9d8ceb08aec0349',
 'original_files/reports/round20_plan.md': '4281480610c65f1cd914d10058ef1c1de8f912c5e7653b0f2932c0e05d1d40bd',
 'original_files/src/round20_experiment.py': 'f3666755840ae1f640678d041bf7501853ad4c3aa11f81779104ddebdc1e6f8c',
 'original_files/src/round20_report.py': '6f226f2b049160f9ace6ff00e582c4bdfb90736de8fdaae0c167f89a76934327'}
HISTORICAL_FILES = frozenset(FAILURE_PREFIX+n for n in
    FAILURE_OUTPUT_NAMES | set(FAILURE_ORIGINAL_FILES) | {'failure_record.json', 'failure_record.sha256'})
PRE_SCORE_RETRY_RECORD = {
 'path': FAILURE_PREFIX+'failure_record.json', 'sha256': FAILURE_RECORD_SHA,
 'status': 'superseded_pre_score_infrastructure_failure',
 'query_price_evaluation_executed': False, 'model_hyperparameters_changed_for_retry': False}
NOTICE = '# R20 development-rejection snapshot\n\nThis is an experiment archive, not a release. The completed R20 development\ncandidate was rejected by the fixed gates. Confirmation/auxiliary price\nfits/evaluations and final fitting were not run; submission_v3.csv remains current.\n\nThe component uses the unchanged R6 geographic view: 95 features, 23 categories,\nraw target y_TWD/1e6, uniform training weights, depth 8 and Plain boosting.\nOnly iterations 2999 to 5998 and learning_rate .065 to .0325 changed.\nThe step-size product does not imply equivalent capacity or less overfitting.\nSaved CBM native learning rate is authoritative; the fitted in-memory value is\nretained separately, and their difference uses the fixed 1e-7 relative tolerance.\n\nThe first 5998-tree fit completed but its reload comparison failed before query\nprice evaluation. Its exact 16 preserved files are copied as historical bytes,\nincluding the explicitly recorded old staging directory. The old source maps\nare not current authority. The same model configuration was refrozen/retrained;\nonly the subsequent completed result is evaluated. Current partial staging is\nstill refused. No historical completion or prediction equality is fabricated.\n\nFive R20 sources, documents, finalized outputs, logs and historical failure bytes\nare copied. Original data, earlier models, four annual geography chains and the\n11 static historical method dependencies stay external and are verified by SHA\nand size. This is not self-contained. All copied originals remain byte-guarded.\n\nCreate requires exactly one CLOSED work-log marker on the same bytes archived.\nVerification reads JSON and hashes binary files; it never deserializes models or\narrays, refits models, recomputes errors, creates predictions, or changes a registry.\nversions.json is creation-time context only; CURRENT_STATUS.md and the registry\nare excluded from permanent copied/external guards. Existing snapshots and\npartial staging are never overwritten or removed.\n'


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
    return (name in SOURCES or name.startswith('outputs/round20/')
            or name in HISTORICAL_FILES or name.startswith('reports/round20_')
            or name == 'README_round20.md')


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
    need(OUT.is_dir(), 'No R20 outputs')
    for area in ('models', 'evaluation'):
        folder=OUT/area
        need(folder.is_dir(), 'Missing current '+area)
        validate_current_stage_names({p.name for p in folder.iterdir()})
    for name in ('final','release'):
        need(not (OUT/name).exists(), 'R20 rejection cannot contain '+name)
    need(not list((ROOT/'reports').glob('.round20-report-*')), 'Partial report staging requires inspection; keep it')
    model=OUT/'models/development'; evaluation=OUT/'evaluation/development'
    need({p.name for p in model.iterdir()} == MODEL_FILES and all(p.is_file() for p in model.iterdir()), 'Incomplete model inventory')
    need({p.name for p in evaluation.iterdir()} == {CONTROL+'.npy',CANDIDATE+'.npy','results.json','results.sha256'}
         and all(p.is_file() for p in evaluation.iterdir()), 'Incomplete evaluation inventory')
    history=ROOT/FAILURE_PREFIX
    need(history.is_dir(), 'Missing preserved pre-score attempt')
    need({rel(p) for p in history.rglob('*') if p.is_file()} == HISTORICAL_FILES,
         'Historical failure inventory differs from exact 16 files')
    validate_current_paths({p.relative_to(OUT).as_posix() for p in OUT.rglob('*')})
    paths=set(OUT.rglob('*')) | set(history.rglob('*'))
    for path in (ROOT/'reports').glob('round20_*'):
        paths.add(path)
        if path.is_dir(): paths.update(path.rglob('*'))
    paths.update(ROOT/name for name in SOURCES)
    if (ROOT/'README_round20.md').exists(): paths.add(ROOT/'README_round20.md')
    files=set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink is not an archive artifact: '+str(path))
        if path.is_file():
            name=rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R20 copy scope')
            if name.startswith('reports/'):
                need(not name.lower().endswith(('.tmp','.part','.lock')), 'Partial report file')
            need(not (path.name.lower().startswith('submission') and path.suffix.lower()=='.csv'), 'No new submission in experiment archive')
            files.add(name)
    need(set(SOURCES).union(DOCS,HISTORICAL_FILES) <= files, 'Required R20 source/document/history missing')
    return files


def rejection_contract(plan, frozen, accepted, result):
    need(plan['round'] == frozen['round'] == accepted['round'] == result['round'] == 20,
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
    need(plan['feature_count']==95 and plan['categorical_count']==23
         and plan['parameters']['random_seed']==42 and plan['parameters']['thread_count']==4
         and plan['parameters']['loss_function']=='MAE'
         and plan['component']=='geography_slow_catboost'
         and plan['sample_weight']=='uniform' and plan['target_mode']=='raw', 'Fixed raw model contract differs')
    validate_parameter_change(plan['baseline_parameters'],plan['parameters'])
    need(plan['changed_parameter_names']==CHANGED_PARAMETERS
         and plan['model_hyperparameter_change']=={'iterations':{'from':2999,'to':5998},'learning_rate':{'from':.065,'to':.0325}}
         and plan['capacity_constraint']=={'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'}
         and plan['native_learning_rate_contract']==LR_CONTRACT, 'Fixed parameter/capacity contract differs')
    need(plan['removed_feature_names']==[] and plan['feature_rules']['removed_feature_names']==[], 'Original 95 features must remain')
    validate_development_decision(result['decision'])
    need(plan['target_formula'] == 'y_TWD / 1e6'
         and plan['inverse_formula'] == 'max(z * 1e6, 0)', 'Wrong target/inverse')
    need(result['fit_rows'] == plan['split_contract']['development']['fit_rows']
         and result['query_rows'] == plan['split_contract']['development']['query_rows'], 'Rows differ')
    for field in ('formulas', 'thresholds'):
        need(result[field] == frozen[field] == plan[field], 'Frozen protocol differs: ' + field)
    need(frozen['parameters'] == plan['parameters'] and result['target_group_weights']
         == frozen['target_group_weights'] == plan['target_group_weights'], 'Frozen parameters/weights differ')


def validate_parameter_change(baseline,params):
    need(set(baseline)==set(params) and baseline['iterations']==2999 and baseline['learning_rate']==.065
         and params['iterations']==5998 and params['learning_rate']==.0325
         and params['depth']==baseline['depth']==8 and params['boosting_type']==baseline['boosting_type']=='Plain'
         and {k for k in baseline if baseline[k]!=params[k]}==set(CHANGED_PARAMETERS),
         'Only the fixed iterations and learning rate may differ from R6')


def validate_learning_rates(saved,fitted):
    for rate in (saved,fitted):
        need(isinstance(rate,(int,float)) and not isinstance(rate,bool) and math.isfinite(rate)
             and abs(rate-.0325)<=1e-7*abs(.0325), 'Native learning rate violates fixed tolerance')
    need(abs(fitted-saved)<=1e-7*abs(saved), 'Fitted/saved learning rates exceed the fixed reload tolerance')


def validate_development_decision(decision):
    keys={'overall_improves_at_least_1pct','weighted_improves_at_least_1pct',
          'quarters_deteriorate_at_most_5pct','large_strata_deteriorate_at_most_5pct'}
    need(set(decision['checks'])==keys and all(type(v) is bool for v in decision['checks'].values()),
         'Fixed development gate inventory differs')
    passed=all(decision['checks'].values())
    need(decision['passed'] is passed and decision['selected_candidate']==(CANDIDATE if passed else CONTROL),
         'Saved gate booleans and selected candidate disagree')


def check_capacity(model,proof):
    capacity={key:model[key] for key in CAPACITY_FIELDS}
    need(capacity['actual_model_depth']==8 and capacity['actual_boosting_type']=='Plain'
         and capacity['actual_grow_policy']=='SymmetricTree' and capacity['actual_trees']==5998,
         'Wrong native CatBoost capacity')
    need(type(capacity['maximum_tree_leaf_count']) is int and 1<=capacity['maximum_tree_leaf_count']<=256,
         'Wrong native leaf-cap evidence')
    need(isinstance(capacity['tree_leaf_counts_sha256'],str)
         and re.fullmatch('[0-9a-f]{64}',capacity['tree_leaf_counts_sha256']), 'Invalid LE-int64 leaf digest')
    validate_learning_rates(capacity['actual_learning_rate'],model['fitted_actual_learning_rate'])
    need(proof['native_model_capacity']==capacity, 'Proof/model saved native capacity differs')
    for value in (model,proof):
        need(value['native_categorical_feature_count']==23
             and value['native_learning_rate_comparison_rtol']==1e-7
             and value['native_learning_rate_comparison_atol']==0.0, 'Native category/LR contract differs')


def check_report(report, plan, accepted, result, model, proof, proof_sha):
    need(report['round'] == 20 and report['accepted'] is False
         and report['candidate'] == CANDIDATE
         and report['currently_recommended_submission'] == 'outputs/submission_v3.csv'
         and report['stopping_reason'] == accepted['stopping_reason']
         and report['deployment_decision'] == 'keep_v3', 'Report decision differs')
    for key in ('new_submission_created', 'historically_untouched', 'public_score_used', 'private_score_known'):
        need(report[key] is False, 'Report policy differs: ' + key)
    need(report['pre_score_retry_record']==PRE_SCORE_RETRY_RECORD, 'Report pre-score retry evidence differs')
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
                'proof_path': 'outputs/round20/audit/development_replay.json', 'proof_sha256': proof_sha,
                'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal'],
                'native_model_capacity':proof['native_model_capacity'],
                'removed_feature_names':proof['removed_feature_names'],
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



def validate_current_stage_names(names):
    need(set(names)=={'development'}, 'Current staging or unauthorized phase must be preserved and inspected')


def validate_current_paths(names):
    for name in names:
        normalized(name)
        need(not any(part.startswith(('.', '_stage_')) for part in name.split('/'))
             and not name.lower().endswith(('.tmp','.part','.lock')),
             'Current incomplete staging must be preserved and inspected: '+name)


def research_expectations(text):
    tree=ast.parse(text)
    nodes=[n for n in tree.body if isinstance(n,ast.Assign) and
           any(isinstance(t,ast.Name) and t.id=='RESEARCH_EXPECTED_SHA' for t in n.targets)]
    need(len(nodes)==1, 'Missing/duplicate static research declaration')
    value=ast.literal_eval(nodes[0].value)
    need(isinstance(value,dict) and len(value)==11, 'Expected eleven static historical method dependencies')
    for name,digest in value.items():
        normalized(name)
        need(re.fullmatch('[0-9a-f]{64}',digest) is not None, 'Invalid static research SHA')
    return value


def validate_research_map(plan,mapping):
    need(plan['research_dependency_sha256']==mapping
         and all(plan['dependency_sha256'].get(k)==v for k,v in mapping.items()), 'Static research dependencies differ')


def check_research(plan,bindings):
    name='src/round20_experiment.py'
    bindings.add(name,plan['source_sha256'][name])
    mapping=research_expectations(safe(name).read_text(encoding='utf-8-sig'))
    validate_research_map(plan,mapping)
    bindings.mapping(mapping)  # Hash only: do not interpret prior model scores.
    return {'frozen_research_dependency_count':11,'dependency_sha256':copy.deepcopy(mapping),
      'historical_source_configuration_metadata_document_evidence_only':True,
      'new_research_JSON_or_receipt_required':False,
      'report_did_not_read_historical_metrics_for_selection':True}


def historical_failure_mapping(record):
    need(record['status']==PRE_SCORE_RETRY_RECORD['status']
         and record['original_directory']=='outputs/round20'
         and record['preserved_directory']==FAILURE_PREFIX.rstrip('/'), 'Wrong historical failure identity')
    need(record['query_price_evaluation_executed'] is False
         and record['full_query_exact_equality_check_reached'] is False
         and record['model_hyperparameters_changed_for_retry'] is False, 'Wrong pre-score retry scope')
    need(record['original_source_sha256']==FAILURE_ORIGINAL_FILES['original_files/src/round20_experiment.py'],
         'Wrong original driver identity')
    source=record['original_output_sha256']
    need(isinstance(source,dict) and set(source)==FAILURE_OUTPUT_NAMES, 'Wrong historical nine-output inventory')
    mapping={}
    for name,digest in source.items():
        normalized(name)
        need(re.fullmatch('[0-9a-f]{64}',digest) is not None, 'Invalid historical output digest')
        mapping[FAILURE_PREFIX+name]=digest  # Rebase old relative paths; never bind live OUT.
    mapping.update({FAILURE_PREFIX+k:v for k,v in FAILURE_ORIGINAL_FILES.items()})
    return mapping


PLAN_RETRY_FIELDS = ('candidate','control','component','parameters','baseline_parameters',
    'changed_parameter_names','removed_feature_names','target_mode','target_formula','inverse_formula',
    'sample_weight','weight_argument_passed_to_pool','feature_count','categorical_count','feature_rules',
    'splits','split_contract','thresholds','target_group_weights','formulas','capacity_constraint',
    'native_learning_rate_contract','model_hyperparameter_change')


def historical_plan_contract(original,current):
    need(original['round']==current['round']==20, 'Historical/current plan round differs')
    need(all(original[key]==current[key] for key in PLAN_RETRY_FIELDS),
         'Pre-score retry changed the model, features, split, blend, or acceptance contract')


def check_history(bindings,plan):
    name=FAILURE_PREFIX+'failure_record.json'
    bindings.add(name,FAILURE_RECORD_SHA)
    record=bindings.authority(name)
    mapping=historical_failure_mapping(record)
    bindings.mapping(mapping)
    original_plan=bindings.authority(FAILURE_PREFIX+'candidate_plan.json')
    historical_plan_contract(original_plan,plan)
    need(original_plan['source_sha256']['src/round20_experiment.py']==record['original_source_sha256'],
         'Historical plan source does not bind the preserved original driver')
    need(len(mapping)==14 and set(mapping)|{name,FAILURE_PREFIX+'failure_record.sha256'}==HISTORICAL_FILES,
         'Preserved 9 outputs plus 5 original files plus 2 record files differ')
    return {'record_path':name,'record_sha256':FAILURE_RECORD_SHA,'historical_file_count':16,
            'historical_original_output_count':9,'historical_original_files_count':5,
            'old_source_maps_used_as_current_authority':False,'current_staging_exception_allowed':False,
            'historical_and_current_plans_share_exact_model_and_validation_contract':True,
            'unchanged_plan_contract_fields':list(PLAN_RETRY_FIELDS),
            'query_price_evaluation_executed':False,'full_query_exact_equality_check_reached':False,
            'model_hyperparameters_changed_for_retry':False}


def collect():
    """Read finalized authorities and hashes. Never inspect npy/model contents."""
    files = inventory()
    b = Bindings()
    work_log_name = 'reports/round20_work_log.md'
    work_log_bytes = safe(work_log_name).read_bytes()
    b.add(work_log_name, closed_work_log_sha256(work_log_bytes))
    authority_paths = {
        'plan': 'outputs/round20/candidate_plan.json',
        'frozen': 'outputs/round20/frozen_candidate.json',
        'accepted': 'outputs/round20/accepted_candidate.json',
        'result': 'outputs/round20/evaluation/development/results.json',
        'model': 'outputs/round20/models/development/metadata.json',
        'proof': 'outputs/round20/audit/development_replay.json',
        'report': 'reports/round20_results.json',
    }
    values = {key: b.authority(name) for key, name in authority_paths.items()}
    plan, frozen, accepted, result, model, proof, report = (values[k] for k in authority_paths)
    digests = {key: b.values[name]['sha256'] for key, name in authority_paths.items()}
    rejection_contract(plan, frozen, accepted, result)
    for value in (frozen, accepted, result, model, proof):
        need(value['plan_sha256'] == digests['plan'], 'Authority refers to another R20 plan')
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
    historical_failure = check_history(b,plan)
    need(report['research_summary'] == research_summary, 'Report research scope differs')
    b.add('src/round20_report.py', REPORT_SOURCE_SHA)
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
    expected_outputs = {'outputs/round20/evaluation/development/' + name + '.npy'
                        for name in (CONTROL, CANDIDATE)}
    need(set(result['output_artifact_sha256']) == expected_outputs, 'Unexpected prediction inventory')
    need(model['round'] == 20 and model['stage'] == 'development'
         and model['component'] == plan['component'] and model['parameters'] == plan['parameters'],
         'Model configuration differs')
    need(model['iterations'] == model['actual_iterations'] == model['actual_trees'] == 5998
         and model['family'] == 'catboost'
         and model['weight_argument_passed_to_pool'] is False
         and model['feature_rules'] == plan['feature_rules']
         and model['feature_count'] == 95 and model['categorical_feature_count'] == 23
         and model['sample_weight'] == 'uniform' and model['target_mode'] == plan['target_mode']
         and model['target_formula'] == plan['target_formula']
         and model['inverse_formula'] == plan['inverse_formula'], 'Model scale/schema differs')
    need(model['baseline_parameters'] == proof['baseline_parameters'] == plan['baseline_parameters']
         and model['changed_parameter_names'] == proof['changed_parameter_names']
         == plan['changed_parameter_names'] == CHANGED_PARAMETERS
         and model['removed_feature_names'] == proof['removed_feature_names'] == plan['removed_feature_names'] == REMOVED_FEATURES,
         'Fixed two-parameter change or original-feature evidence differs')
    validate_parameter_change(model['baseline_parameters'],model['parameters'])
    check_capacity(model, proof)
    need(model['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and model['feature_rules']['geo_encoder_rule_training_fields_are_baseline_provenance_only'] is True
         and model['feature_rules']['iterations']==5998
         and model['feature_rules']['learning_rate']==.0325
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
    need(proof['round'] == 20 and proof['stage'] == 'development' and proof['status'] == 'passed'
         and proof['candidate_accepted_in_this_stage'] is False and proof['saved_raw_z_bitwise_equal'] is True,
         'Independent proof does not certify this rejection')
    need(proof['independent_metrics'] == result['metrics']
         and proof['independent_decision'] == result['decision']
         and proof['raw_z_diagnostics'] == model['raw_z_diagnostics'], 'Proof/report payload mismatch')
    for key in ('fit_performed', 'new_window_scored', 'public_score_used', 'historically_untouched'):
        need(proof[key] is False, 'Unexpected independent review behavior: ' + key)
    need(proof['model_family'] == 'catboost' and proof['model_format'] == 'native_CatBoost_cbm'
         and proof['raw_z_units'] == 'million_TWD' and proof['actual_iterations'] == 5998
         and proof['feature_count'] == 95 and proof['categorical_count'] == 23,
         'Independent raw geographic CatBoost schema differs')
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
    b.add('src/round20_verify_backtest.py', proof['source_sha256'])
    check_report(report, plan, accepted, result, model, proof, digests['proof'])
    # These three source maps are authoritative; verify that the independent
    # replay actually bound the artifacts it says it replayed, without loading them.
    required_proof = {authority_paths[k] for k in ('plan', 'frozen', 'accepted', 'result', 'model')}
    required_proof.update('outputs/round20/models/development/' + name
                          for name in MODEL_FILES)
    required_proof.update(expected_outputs)
    need(required_proof <= set(proof['source_input_sha256']), 'Replay input closure is incomplete')
    required_report = set(required_proof) | {authority_paths['accepted'], authority_paths['proof'],
                                           'src/round20_report.py'}
    need(required_report <= set(report['source_input_sha256']), 'Report input closure is incomplete')
    # Attempts are copied, not interpreted as current source bindings.
    for name in sorted(files):
        b.add(name)
        if name.endswith('.sha256'):
            path = safe(name)
            target = path.with_suffix('.json')
            need(target.is_file(), 'Historical/current receipt has no JSON: ' + name)
            locked(target)  # Byte receipt only; do not traverse historical source maps.
    need(inventory() == files, 'R20 inventory changed during collection')
    need(all(not copy_scope(name) or name in files for name in b.values), 'Owned dependency omitted from copy inventory')
    copied = {name: b.values[name].copy() for name in sorted(files)}
    external = {name: item.copy() for name, item in sorted(b.values.items()) if name not in copied}
    need(not MUTABLE.intersection(set(copied) | set(external)), 'Mutable entry point became permanent')
    registry = safe('runs/versions.json')
    context = {'path': 'runs/versions.json', 'sha256': sha(registry), 'bytes': registry.stat().st_size,
               'plan_time_sha256': plan['registry_sha256_at_plan'],
               'authority_observed_sha256': sorted(b.mutable_declarations.get('runs/versions.json', set())),
               'verification_policy': 'creation-time observation only; never a future external guard'}
    state = {'round': 20, 'is_release': False, 'accepted_backtest': False,
             'candidate': CANDIDATE, 'selected_candidate': CONTROL,
             'stopping_reason': accepted['stopping_reason'], 'stage_status': accepted['stage_status'],
             'recommended_version': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
             'new_submission_created': False, 'registry_modified': False,
             'plan_sha256': digests['plan'], 'frozen_candidate_sha256': digests['frozen'],
             'accepted_candidate_sha256': digests['accepted'], 'development_result_sha256': digests['result'],
             'independent_replay_sha256': digests['proof'], 'report_json_sha256': digests['report'],
             'source_path': 'src/archive_round20.py', 'source_sha256': sha(__file__),
             'property_labels_or_prediction_values_read_by_archiver': False,
             'array_or_model_deserialization_performed': False, 'models_refitted_or_MAE_recomputed': False,
             'index_identity_uses_saved_contract_and_replay_not_array_deserialization': True,
             'historically_untouched': False, 'historical_attempts_kept_without_current_source_enforcement': True,
             'registry_context': context, 'current_status_is_not_bound': True,
             'research_dependency_sha256': plan['research_dependency_sha256'],
             'research_is_static_historical_method_evidence': True,
             'new_research_folder_or_receipt_expected': False, 'research_summary': research_summary,
             'historical_pre_score_failure': historical_failure}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round20_development_rejection_experiments', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 20 and state['is_release'] is False and state['accepted_backtest'] is False
         and state['selected_candidate'] == CONTROL and state['stopping_reason'] == 'development_price_gates_failed'
         and state['recommended_version'] == 'v3' and state['new_submission_created'] is False,
         'Snapshot does not describe the finalized development rejection')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copied and external paths overlap')
    names = set(copied) | set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names} & {n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, HISTORICAL_FILES) <= set(copied) and all(copy_scope(n) for n in copied),
         'Wrong R20 copy inventory')
    need(all(not copy_scope(n) for n in external), 'Owned R20 artifact left external')
    need(manifest['copied_file_count'] == len(copied)
         and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(item['bytes'] for item in copied.values()),
         'Manifest count/size mismatch')
    for name, item in {**copied, **external}.items():
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid file identity: ' + name)
    for key, name in [('plan_sha256', 'outputs/round20/candidate_plan.json'),
                      ('frozen_candidate_sha256', 'outputs/round20/frozen_candidate.json'),
                      ('accepted_candidate_sha256', 'outputs/round20/accepted_candidate.json'),
                      ('development_result_sha256', 'outputs/round20/evaluation/development/results.json'),
                      ('independent_replay_sha256', 'outputs/round20/audit/development_replay.json'),
                      ('report_json_sha256', 'reports/round20_results.json'),
                      ('source_sha256', 'src/archive_round20.py')]:
        need(state[key] == copied[name]['sha256'], 'Manifest authority identity mismatch: ' + key)
    history=state['historical_pre_score_failure']
    need(history['record_sha256']==FAILURE_RECORD_SHA and history['historical_file_count']==16
         and history['old_source_maps_used_as_current_authority'] is False
         and history['current_staging_exception_allowed'] is False
         and history['historical_and_current_plans_share_exact_model_and_validation_contract'] is True
         and history['unchanged_plan_contract_fields']==list(PLAN_RETRY_FIELDS)
         and copied[FAILURE_PREFIX+'failure_record.json']['sha256']==FAILURE_RECORD_SHA,
         'Historical failure archive binding differs')
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
        need(inventory() == set(copied), 'Original R20 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}


def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round20_experiments_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round20_experiments_staging_' + uuid.uuid4().hex)
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
    manifest = {'kind': 'round20_development_rejection_experiments',
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
    """Complete bounded R20 contract checks; no actual authority/data/archive reads."""
    checks=work_log_self_check()
    def check(name,value):
        need(bool(value),'Synthetic check failed: '+name); checks.append(name)
    def reject(name,op):
        try: op()
        except (RuntimeError,KeyError,TypeError,ValueError): checks.append(name)
        else: raise AssertionError('Expected rejection: '+name)
    baseline={'iterations':2999,'learning_rate':.065,'depth':8,'random_seed':42,
      'thread_count':4,'loss_function':'MAE','boosting_type':'Plain','max_ctr_complexity':2,'one_hot_max_size':8}
    params={**baseline,'iterations':5998,'learning_rate':.0325}
    plan={'round':20,'candidate':CANDIDATE,'control':CONTROL,'historically_untouched':False,
      'feature_count':95,'categorical_count':23,'parameters':dict(params),'baseline_parameters':dict(baseline),
      'changed_parameter_names':list(CHANGED_PARAMETERS),'removed_feature_names':[],
      'feature_rules':{'removed_feature_names':[]},'sample_weight':'uniform','target_mode':'raw',
      'weight_argument_passed_to_pool':False,
      'capacity_constraint':{'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'},
      'native_learning_rate_contract':dict(LR_CONTRACT),
      'model_hyperparameter_change':{'iterations':{'from':2999,'to':5998},'learning_rate':{'from':.065,'to':.0325}},
      'component':'geography_slow_catboost','target_formula':'y_TWD / 1e6','inverse_formula':'max(z * 1e6, 0)',
      'formulas':{'synthetic_only':True},'thresholds':{'synthetic_only':True},'target_group_weights':{'seen':1.},
      'splits':{k:['start','end'] for k in STAGES},
      'split_contract':{k:{'fit_rows':5,'query_rows':3} for k in STAGES}}
    frozen={'round':20,'candidate':CANDIDATE,'passed':False,'later_stages_allowed':False,
      'selected_candidate':CONTROL,'historically_untouched':False,
      **{k:copy.deepcopy(plan[k]) for k in ('formulas','thresholds','parameters','target_group_weights')}}
    accepted={'round':20,'candidate':CANDIDATE,'accepted':False,'selected_candidate':CONTROL,
      'stopping_reason':'development_price_gates_failed','deployment_decision':'keep_v3',
      'stage_status':{k:'price_evaluated' if k=='development' else 'not_executed_after_development_rejection' for k in STAGES},
      'unevaluated_stages':list(STAGES[1:]),'stage_result_sha256':{'development':'a'*64},
      'new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv',
      'no_reselection_or_parameter_change':True,'historically_untouched':False}
    decision={'passed':False,'selected_candidate':CONTROL,'checks':{
      'overall_improves_at_least_1pct':False,'weighted_improves_at_least_1pct':False,
      'quarters_deteriorate_at_most_5pct':True,'large_strata_deteriorate_at_most_5pct':True}}
    result={'round':20,'stage':'development','evaluated_candidates':[CONTROL,CANDIDATE],
      'decision':copy.deepcopy(decision),'fit_rows':5,'query_rows':3,'historically_untouched':False,
      'metrics':{'synthetic_saved_value':1.},**{k:copy.deepcopy(plan[k]) for k in ('formulas','thresholds','target_group_weights')}}
    rejection_contract(plan,frozen,accepted,result); checks.append('raw95_23_fixed_two_parameter_development_rejection')
    original_plan=copy.deepcopy(plan);original_plan['source_sha256']={'src/round20_experiment.py':'a'*64}
    retried_plan=copy.deepcopy(plan);retried_plan['source_sha256']={'src/round20_experiment.py':'b'*64}
    original_plan['created_at_utc']='old';retried_plan['created_at_utc']='new'
    historical_plan_contract(original_plan,retried_plan);checks.append('retry_unchanged_recipe_with_corrected_source_and_timestamp_accepted')
    for key in ('parameters','feature_rules','splits','split_contract','thresholds','target_group_weights','formulas'):
        bad=copy.deepcopy(retried_plan);bad[key]={'synthetic_changed':True}
        reject('retry_changed_'+key+'_rejected',lambda bad=bad:historical_plan_contract(original_plan,bad))
    for key,value in [('feature_count',91),('categorical_count',19),('changed_parameter_names',[]),
                      ('removed_feature_names',['project_name']),('sample_weight','effective_area / 100')]:
        bad=copy.deepcopy(plan);bad[key]=value
        reject('wrong_plan_'+key,lambda bad=bad:rejection_contract(bad,frozen,accepted,result))
    for key,value in [('depth',6),('iterations',2999),('learning_rate',.065),('boosting_type','Ordered'),('max_ctr_complexity',1)]:
        bad=dict(params);bad[key]=value
        reject('wrong_or_third_changed_parameter_'+key,lambda bad=bad:validate_parameter_change(baseline,bad))
    bad=copy.deepcopy(plan);bad['native_learning_rate_contract']['rtol']=1e-5
    reject('widened_LR_tolerance_rejected',lambda:rejection_contract(bad,frozen,accepted,result))
    bad=copy.deepcopy(result);bad['decision']['checks']=dict.fromkeys(decision['checks'],True)
    reject('all_saved_gates_pass_cannot_mean_reject',lambda:rejection_contract(plan,frozen,accepted,bad))
    bad=copy.deepcopy(result);bad['decision']['checks'].pop('overall_improves_at_least_1pct')
    reject('missing_gate_rejected',lambda:rejection_contract(plan,frozen,accepted,bad))
    bad=copy.deepcopy(frozen);bad['later_stages_allowed']=True
    reject('later_authorization_after_rejection_rejected',lambda:rejection_contract(plan,bad,accepted,result))
    bad=copy.deepcopy(accepted);bad['stage_result_sha256']['confirmation']='b'*64
    reject('later_result_rejected',lambda:rejection_contract(plan,frozen,bad,result))
    model={'family':'catboost','component':plan['component'],'target_mode':'raw','iterations':5998,
      'actual_iterations':5998,'actual_trees':5998,'feature_count':95,'categorical_feature_count':23,
      'sample_weight':'uniform','weight_argument_passed_to_pool':False,'baseline_parameters':dict(baseline),
      'changed_parameter_names':list(CHANGED_PARAMETERS),'parameters':dict(params),'removed_feature_names':[],
      'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_boosting_type':'Plain',
      'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'b'*64,
      'actual_learning_rate':.03249999881,'fitted_actual_learning_rate':.032499998807907104,
      'target_formula':plan['target_formula'],'inverse_formula':plan['inverse_formula'],
      'raw_z_diagnostics':{'synthetic':True},'runner_elapsed_seconds':1.,'native_categorical_feature_count':23,
      'native_learning_rate_comparison_rtol':1e-7,'native_learning_rate_comparison_atol':0.,
      'geo_encoder_rule_training_fields_are_baseline_provenance_only':True}
    proof={'replays':{'synthetic':True},'saved_raw_z_bitwise_equal':True,'removed_feature_names':[],
      'native_model_capacity':{k:model[k] for k in CAPACITY_FIELDS},'native_categorical_feature_count':23,
      'native_learning_rate_comparison_rtol':1e-7,'native_learning_rate_comparison_atol':0.,
      'independent_fit_state_evidence':{'fit_rows':5,'base_feature_count':95,'feature_count':95,
        'categorical_count':23,'removed_feature_names':[],'fit_state_reconstructed_independently':True,
        'fit_model_feature_values_sha256':'c'*64}}
    check_capacity(model,proof);checks.append('native7_saved_LR_and_fitted_float32_representation_accepted')
    for name,saved,fitted in [('requested',.0325,.0325),('float32',.032499998807907104,.03249999881)]:
        validate_learning_rates(saved,fitted);checks.append('fixed_tolerance_'+name+'_accepted')
    for name,saved,fitted in [('old_rate',.065,.0325),('old_fitted_rate',.0325,.065),
                            ('NaN',float('nan'),.0325),('infinity',.0325,float('inf'))]:
        reject('native_LR_'+name+'_rejected',lambda saved=saved,fitted=fitted:validate_learning_rates(saved,fitted))
    for key,value in [('actual_trees',2999),('actual_model_depth',6),('maximum_tree_leaf_count',257),
                      ('actual_boosting_type','Ordered'),('native_categorical_feature_count',19)]:
        bad=copy.deepcopy(model);bad[key]=value
        reject('native_'+key+'_rejected',lambda bad=bad:check_capacity(bad,proof))
    bad=copy.deepcopy(proof);bad['native_model_capacity']['actual_learning_rate']=.0325
    reject('proof_saved_LR_must_match_saved_metadata_exactly',lambda:check_capacity(model,bad))
    mapping={f'old/method_{i}.json':'a'*64 for i in range(11)}
    check('static_research_AST_literal_only',research_expectations('RESEARCH_EXPECTED_SHA = '+repr(mapping))==mapping)
    research_plan={'research_dependency_sha256':dict(mapping),'dependency_sha256':dict(mapping)}
    validate_research_map(research_plan,mapping);checks.append('eleven_research_subset_closure')
    bad=copy.deepcopy(research_plan);bad['dependency_sha256'].pop(next(iter(mapping)))
    reject('research_missing_dependency_rejected',lambda:validate_research_map(bad,mapping))
    reject('research_nonliteral_not_executed',lambda:research_expectations('RESEARCH_EXPECTED_SHA = dangerous()'))
    record={'status':PRE_SCORE_RETRY_RECORD['status'],'original_directory':'outputs/round20',
      'preserved_directory':FAILURE_PREFIX.rstrip('/'),'query_price_evaluation_executed':False,
      'full_query_exact_equality_check_reached':False,'model_hyperparameters_changed_for_retry':False,
      'original_source_sha256':FAILURE_ORIGINAL_FILES['original_files/src/round20_experiment.py'],
      'original_output_sha256':dict.fromkeys(FAILURE_OUTPUT_NAMES,'d'*64)}
    history=historical_failure_mapping(record)
    check('historical_nine_outputs_rebased_and_five_old_sources_bound',len(history)==14
          and all(n.startswith(FAILURE_PREFIX) for n in history)
          and not any(n.startswith('outputs/round20/') or n.startswith('src/') for n in history))
    check('historical_exact16_copy_scope',len(HISTORICAL_FILES)==16 and all(copy_scope(n) for n in HISTORICAL_FILES)
          and all(len(d)==64 for d in FAILURE_ORIGINAL_FILES.values()))
    bad=copy.deepcopy(record);bad['original_output_sha256']['../candidate_plan.json']='d'*64
    reject('historical_parent_escape_rejected',lambda:historical_failure_mapping(bad))
    bad=copy.deepcopy(record);bad['original_output_sha256']['outputs/round20/candidate_plan.json']='d'*64
    reject('historical_live_path_alias_rejected',lambda:historical_failure_mapping(bad))
    bad=copy.deepcopy(record);bad['original_source_sha256']='f'*64
    reject('old_driver_cannot_be_rebound_to_new_driver',lambda:historical_failure_mapping(bad))
    bad=copy.deepcopy(record);bad['query_price_evaluation_executed']=True
    reject('pre_score_history_cannot_claim_a_price_evaluation',lambda:historical_failure_mapping(bad))
    validate_current_stage_names({'development'});validate_current_paths({'models/development/model.cbm','logs/development_fit.log'})
    checks.append('complete_current_development_only_inventory_accepted')
    for names in ({'development','.development-any'}, {'development','confirmation'},set()):
        reject('current_stage_inventory_rejected_'+repr(sorted(names)),lambda names=names:validate_current_stage_names(names))
    for name in ('models/.development-old/model.cbm','.plan-any/candidate_plan.json','logs/file.part'):
        reject('current_partial_path_rejected_'+name,lambda name=name:validate_current_paths({name}))
    check('historical_preserved_staging_only_explicit_namespace',copy_scope(FAILURE_PREFIX+FAILURE_STAGE+'model.cbm')
          and not copy_scope(FAILURE_PREFIX+'models/.development-other/model.cbm'))
    check('five_owned_sources_and_seven_docs_copyable',len(SOURCES)==5 and len(DOCS)==7 and all(copy_scope(n) for n in SOURCES+DOCS))
    check('old_data_geo_sources_and_mutable_paths_external_or_context',not any(copy_scope(n) for n in
      ('src/house_features.py','src/round6_geo.py','data_cache/train_features.pkl',
       'data_external/geography/taiwan-220101.osm.pbf','outputs/round6/geography/final/manifest.json',
       'CURRENT_STATUS.md','runs/versions.json','src/round20_unowned.py')))
    report={'round':20,'candidate':CANDIDATE,'accepted':False,'currently_recommended_submission':'outputs/submission_v3.csv',
      'stopping_reason':accepted['stopping_reason'],'deployment_decision':'keep_v3','new_submission_created':False,
      'historically_untouched':False,'public_score_used':False,'private_score_known':False,
      'report_only_reads_saved_JSON_and_hashes_binary_inputs':True,'pre_score_retry_record':dict(PRE_SCORE_RETRY_RECORD),'stages':[]}
    for stage in STAGES:
        row={'stage':stage,'status':accepted['stage_status'][stage],'dates':plan['splits'][stage],
          'fit_rows':5,'query_rows':3,'metrics':None,'decision':None,'model':None,'independent_review':None}
        if stage=='development':
            row.update(metrics=result['metrics'],decision=result['decision'],model={k:model[k] for k in MODEL_SUMMARY_FIELDS},
              independent_review={'proof_path':'outputs/round20/audit/development_replay.json','proof_sha256':'a'*64,
                'replays':proof['replays'],'raw_z_bitwise_equal':True,'native_model_capacity':proof['native_model_capacity'],
                'removed_feature_names':[],'independent_fit_state_evidence':proof['independent_fit_state_evidence']})
        report['stages'].append(row)
    check_report(report,plan,accepted,result,model,proof,'a'*64);checks.append('report_native7_fittedLR_and_historical_record_match')
    bad=copy.deepcopy(report);bad['stages'][1]['metrics']={}
    reject('report_later_score_invention_rejected',lambda:check_report(bad,plan,accepted,result,model,proof,'a'*64))
    bad=copy.deepcopy(report);bad['pre_score_retry_record']['query_price_evaluation_executed']=True
    reject('report_historical_score_invention_rejected',lambda:check_report(bad,plan,accepted,result,model,proof,'a'*64))
    bad=copy.deepcopy(report);bad['stages'][0]['model']['fitted_actual_learning_rate']=.0325
    reject('report_fitted_diagnostic_must_match_metadata',lambda:check_report(bad,plan,accepted,result,model,proof,'a'*64))
    binding=Bindings(); binding.declare('src/a.py','a'*64)
    binding.values['src/a.py']={'sha256':'a'*64,'bytes':0}  # Simulate a completed add without reading a file.
    reject('authority_hash_conflict_rejected',lambda:binding.declare('src/a.py','b'*64))
    reject('authority_case_alias_rejected',lambda:binding.declare('src/A.py','a'*64))
    check('mutable_registry_declaration_is_context_only',binding.declare('runs/versions.json','c'*64) is False
          and 'runs/versions.json' not in binding.values)
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
