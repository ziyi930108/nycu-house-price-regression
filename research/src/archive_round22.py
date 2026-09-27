"""Archive the finalized R22 development rejection; stdlib and byte hashes only.
Explicit --create / --verify / --self-check. No model or array deserialization,
training, scoring, release, registry write, cleanup, or old-source imports.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, math, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round22'
DEST = ROOT / 'runs/round22_experiments'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, CANDIDATE = 'v3_control', 'huber_catboost_geography_blend'
SOURCES = ('src/round22_huber_geo.py', 'src/round22_experiment.py',
           'src/round22_verify_backtest.py', 'src/round22_report.py',
           'src/archive_round22.py', 'src/round22_huber_feasibility.py', 'src/round22_report_qa.py')
DOCS = ('reports/round22_plan.md', 'reports/round22_method_review.md',
        'reports/round22_pipeline_review.md', 'reports/round22_work_log.md',
        'reports/round22_report.md', 'reports/round22_results.json', 'reports/round22_results.sha256')
REMOVED_FEATURES = []
ADDED_FEATURES = []
CHANGED_PARAMETERS = ['loss_function']
LR_CONTRACT = {'expected': .065, 'rtol': 1e-7, 'atol': 0.0}
FEASIBILITY_PATH = 'outputs/round22_research/huber_feasibility.json'
METHOD_SOURCES_PATH = 'outputs/round22_research/method_sources.json'
RESEARCH_FILES = (FEASIBILITY_PATH, 'outputs/round22_research/huber_feasibility.sha256',
    METHOD_SOURCES_PATH, 'outputs/round22_research/method_sources.sha256',
    'outputs/round22_research/mae_control.cbm', 'outputs/round22_research/huber_candidate.cbm')
QA_PATH = 'outputs/round22/audit/report_artifact_qa.json'
NATIVE_OBJECTIVE_CONTRACT = {'loss_function': 'Huber:delta=1', 'eval_metric': 'MAE',
    'leaf_estimation_method': 'Newton', 'leaf_estimation_iterations': 1, 'boost_from_average': False}
IMPLICIT_NATIVE_POLICY_CHANGES = {'boost_from_average': {'from': True, 'to': False},
    'leaf_estimation_method': {'from': 'Exact', 'to': 'Newton'}}
IMPLICIT_NATIVE_DEFAULT_DIFFERENCES = {key: {'baseline': value['from'], 'candidate': value['to']}
    for key,value in IMPLICIT_NATIVE_POLICY_CHANGES.items()}

MODEL_FILES = {'model.cbm', 'feature_state.json', 'predictions.npy', 'raw_z.npy',
               'fit_indices.npy', 'query_indices.npy', 'metadata.json', 'metadata.sha256'}
CAPACITY_FIELDS = ('actual_model_depth', 'actual_grow_policy', 'actual_trees',
                  'maximum_tree_leaf_count', 'tree_leaf_counts_sha256', 'actual_boosting_type',
                  'actual_learning_rate', 'actual_loss_function', 'actual_eval_metric',
                  'actual_leaf_estimation_method', 'actual_leaf_estimation_iterations', 'actual_boost_from_average')
MODEL_SUMMARY_FIELDS = ('family', 'component', 'target_mode', 'iterations', 'actual_iterations',
    'parameters', 'baseline_parameters', 'changed_parameter_names', 'removed_feature_names', 'added_feature_names',
    'baseline_feature_count', 'original95_preserved',
    'native_objective_contract', 'implicit_native_default_differences', 'raw_z_units',
    'feature_count', 'categorical_feature_count', 'sample_weight', 'weight_argument_passed_to_pool',
    'target_formula', 'inverse_formula', 'raw_z_diagnostics', 'runner_elapsed_seconds',
    'fitted_actual_learning_rate', 'native_categorical_feature_count',
    'native_learning_rate_comparison_rtol', 'native_learning_rate_comparison_atol',
    'geo_encoder_rule_training_fields_are_baseline_provenance_only') + CAPACITY_FIELDS
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = '# R22 development-rejection snapshot\n\nThis is an experiment archive, not a release. Only development was completed;\nits fixed gates rejected the candidate. Confirmation, auxiliary and final\nprice fits/evaluations were not run. The recommendation remains\noutputs/submission_v3.csv; no new CSV or registry update is created here.\n\nThe original R6 95 features / 23 categories are unchanged. Passed parameters\nchange only loss_function: MAE to Huber:delta=1. Target y_TWD/1e6, uniform Pool,\n2999 trees, learning rate .065, depth8 and Plain remain fixed. Inherited native\ndefaults also change with the loss: leaf estimation Exact to Newton and\nboost_from_average true to false; one leaf estimation iteration is verified.\nNative metadata/proof verifies twelve fields, 23 categorical positions, and\nthe fixed LR tolerance. Delta one is one million TWD; a different objective\nneed not minimize MAE. Original X87 reference/scoring remains unchanged.\n\nSeven owned sources, all round22 outputs and research, and round22 reports are\ncopied. Research includes two artificial two-tree CBMs, feasibility JSON and\nreceipt, method_sources JSON and receipt, the recipe and logs. Artificial CBMs\nare byte evidence, not the formal 2999-tree house-price model. The complete\n27 frozen research identities and nested source/input/evidence/model hashes\nare checked. Historical methods are external dependencies unless owned here.\n\nOriginal data, old models and all four annual geography chains remain external\nSHA/size guards; this is not self-contained. No arrays/models are deserialized,\nno models are refitted, and no price errors are computed. CLOSED is required\nas exactly one status line in the same work-log bytes archived. Current staging\nis refused and preserved; snapshots and files are never overwritten/deleted.\nversions.json is creation-time context only; it and CURRENT_STATUS.md never\nbecome permanent verification guards. Experiments do not publish a release.\n'



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
    return (name in SOURCES or name.startswith('outputs/round22/')
            or name.startswith('outputs/round22_research/')
            or name.startswith('reports/round22_') or name == 'README_round22.md')


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
    need(OUT.is_dir(), 'No R22 outputs')
    for area in ('models', 'evaluation'):
        folder=OUT/area
        need(folder.is_dir(), 'Missing current '+area)
        validate_current_stage_names({p.name for p in folder.iterdir()})
    for name in ('final','release'):
        need(not (OUT/name).exists(), 'R22 rejection cannot contain '+name)
    need(not list((ROOT/'reports').glob('.round22-report-*')), 'Partial report staging requires inspection; keep it')
    model=OUT/'models/development'; evaluation=OUT/'evaluation/development'
    need({p.name for p in model.iterdir()} == MODEL_FILES and all(p.is_file() for p in model.iterdir()), 'Incomplete model inventory')
    need({p.name for p in evaluation.iterdir()} == {CONTROL+'.npy',CANDIDATE+'.npy','results.json','results.sha256'}
         and all(p.is_file() for p in evaluation.iterdir()), 'Incomplete evaluation inventory')
    research = ROOT/'outputs/round22_research'
    need(research.is_dir(), 'Missing owned R22 synthetic research')
    validate_current_paths({p.relative_to(OUT).as_posix() for p in OUT.rglob('*')})
    validate_current_paths({p.relative_to(research).as_posix() for p in research.rglob('*')})
    paths=set(OUT.rglob('*')) | set(research.rglob('*'))
    for path in (ROOT/'reports').glob('round22_*'):
        paths.add(path)
        if path.is_dir(): paths.update(path.rglob('*'))
    paths.update(ROOT/name for name in SOURCES)
    if (ROOT/'README_round22.md').exists(): paths.add(ROOT/'README_round22.md')
    files=set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink is not an archive artifact: '+str(path))
        if path.is_file():
            name=rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R22 copy scope')
            if name.startswith('reports/'):
                need(not name.lower().endswith(('.tmp','.part','.lock')), 'Partial report file')
            need(not (path.name.lower().startswith('submission') and path.suffix.lower()=='.csv'), 'No new submission in experiment archive')
            files.add(name)
    need(set(SOURCES).union(DOCS,RESEARCH_FILES,{QA_PATH,QA_PATH[:-5]+'.sha256'}) <= files, 'Required R22 source/document/research missing')
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
    need(baseline['loss_function']=='MAE' and baseline['eval_metric']=='MAE'
         and baseline['iterations']==2999 and baseline['learning_rate']==.065
         and baseline['depth']==8 and baseline['boosting_type']=='Plain'
         and params==dict(baseline,loss_function='Huber:delta=1')
         and sorted(k for k in params if params[k]!=baseline[k])==['loss_function'],
         'Only the passed loss_function may differ from R6')
    need(not any(k in params for k in ('leaf_estimation_method','leaf_estimation_iterations','boost_from_average')),
         'Do not turn automatic defaults into additional passed parameters')


def validate_objective_contract(value):
    need(isinstance(value,dict) and set(value)==set(NATIVE_OBJECTIVE_CONTRACT), 'Native objective contract fields differ')
    for key,expected in NATIVE_OBJECTIVE_CONTRACT.items():
        need(type(value[key]) is type(expected) and value[key]==expected, 'Native objective contract differs: '+key)


def validate_huber_rules(rules):
    need(rules['feature_count']==95 and rules['categorical_feature_count']==23
         and rules['baseline_feature_count']==95 and rules['original95_preserved'] is True
         and rules['existing_95_values_dtypes_order_and_index_unchanged'] is True
         and rules['removed_feature_names']==[] and rules['added_feature_names']==[]
         and rules['changed_parameter_names']==['loss_function'], 'Exact original95 schema required')
    validate_objective_contract(rules['native_objective_contract'])
    need(rules['baseline_loss_function']=='MAE' and rules['loss_function']=='Huber:delta=1'
         and rules['eval_metric']=='MAE' and rules['huber_delta_in_model_units']==1.
         and rules['huber_delta_TWD']==1e6 and rules['raw_z_units']=='million_TWD'
         and rules['implicit_native_default_differences']==IMPLICIT_NATIVE_DEFAULT_DIFFERENCES
         and rules['native_defaults_explicitly_passed'] is False
         and rules['same_passed_parameters_does_not_imply_same_native_defaults'] is True,
         'Huber loss/default policy rules differ')


def rejection_contract(plan, frozen, accepted, result):
    need(plan['round'] == frozen['round'] == accepted['round'] == result['round'] == 22,
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
         and plan['parameters']['loss_function']=='Huber:delta=1' and plan['parameters']['eval_metric']=='MAE'
         and plan['component']=='geography_huber_catboost'
         and plan['sample_weight']=='uniform' and plan['target_mode']=='raw'
         and plan['weight_argument_passed_to_pool'] is False, 'Fixed raw model contract differs')
    validate_parameter_change(plan['baseline_parameters'],plan['parameters'])
    need(plan['changed_parameter_names']==CHANGED_PARAMETERS
         and plan['model_hyperparameter_change']=={'loss_function':{'from':'MAE','to':'Huber:delta=1'}}
         and plan['capacity_constraint']=={'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'}
         and plan['native_learning_rate_contract']==LR_CONTRACT, 'Fixed parameter/capacity contract differs')
    need(plan['removed_feature_names']==[] and plan['feature_rules']['removed_feature_names']==[], 'Original 95 features must remain')
    need(plan['added_feature_names']==[], 'No added model features allowed')
    validate_huber_rules(plan['feature_rules'])
    validate_objective_contract(plan['native_objective_contract'])
    need(plan['implicit_native_policy_changes']==IMPLICIT_NATIVE_POLICY_CHANGES, 'Loss-dependent default contract differs')
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
    validate_objective_contract({key:capacity['actual_'+key] for key in NATIVE_OBJECTIVE_CONTRACT})
    need(proof['native_model_capacity']==capacity, 'Proof/model saved native capacity differs')
    for value in (model,proof):
        need(value['native_categorical_feature_count']==23
             and value['native_learning_rate_comparison_rtol']==1e-7
             and value['native_learning_rate_comparison_atol']==0.0, 'Native category/LR contract differs')


def check_report(report, plan, accepted, result, model, proof, proof_sha):
    need(report['round'] == 22 and report['accepted'] is False
         and report['candidate'] == CANDIDATE
         and report['currently_recommended_submission'] == 'outputs/submission_v3.csv'
         and report['stopping_reason'] == accepted['stopping_reason']
         and report['deployment_decision'] == 'keep_v3', 'Report decision differs')
    for key in ('new_submission_created', 'historically_untouched', 'public_score_used', 'private_score_known'):
        need(report[key] is False, 'Report policy differs: ' + key)
    need('pre_score_retry_record' not in report, 'R22 must not inherit an R20 retry record')
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
                'proof_path': 'outputs/round22/audit/development_replay.json', 'proof_sha256': proof_sha,
                'replays': proof['replays'], 'raw_z_bitwise_equal': proof['saved_raw_z_bitwise_equal'],
                'native_model_capacity':proof['native_model_capacity'],
                'removed_feature_names':proof['removed_feature_names'],
                'added_feature_names':proof['added_feature_names'],
                'native_objective_contract':proof['native_objective_contract'],
                'implicit_native_policy_changes':proof['implicit_native_policy_changes'],
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
    need(set(found)=={'RESEARCH_FILES','RESEARCH_EXPECTED_SHA'},'Missing literal R22 research declarations')
    seeds,expected=found['RESEARCH_FILES'],found['RESEARCH_EXPECTED_SHA']
    need(isinstance(seeds,(tuple,list)) and len(seeds)>0 and len(set(seeds))==len(seeds)
         and isinstance(expected,dict) and set(seeds)==set(expected), 'Research seed and expected map inventories differ')
    return {name:expected[name] for name in seeds}


def validate_research(research,dependencies,declared,feasibility,method_sources):
    """Only saved artificial compatibility and source history; no property data."""
    need(isinstance(research,dict) and research and research==declared, 'Frozen research differs from literal driver map')
    for path in (FEASIBILITY_PATH,METHOD_SOURCES_PATH):
        need(path in research and path[:-5]+'.sha256' in research, 'Research JSON and receipt must be frozen')
    need(all(dependencies.get(k)==v for k,v in research.items()), 'Research outside frozen dependencies')
    expected=dict(research)
    def add(name,digest):
        need(isinstance(name,str) and isinstance(digest,str) and len(digest)==64
             and all(c in '0123456789abcdef' for c in digest), 'Invalid research identity')
        need(name not in expected or expected[name]==digest, 'Conflicting compatibility provenance')
        expected[name]=digest
    for name,digest in research.items():add(name,digest)
    need(feasibility['round']==22 and feasibility['status']=='synthetic_GPU_compatibility_passed'
         and feasibility['synthetic_only'] is True and feasibility['synthetic_model_count']==2
         and feasibility['property_MAE_computed'] is False
         and feasibility['property_X_y_models_or_predictions_loaded'] is False, 'Wrong artificial compatibility scope')
    need(feasibility['declared_candidate_loss']=='Huber:delta=1' and feasibility['declared_delta_TWD']==1e6
         and feasibility['candidate_passed_parameter_changes']==CHANGED_PARAMETERS
         and feasibility['feasibility_overrides']=={'iterations':2,'verbose':False}, 'Synthetic objective/scope changed')
    baseline=feasibility['baseline_parameters'];params=feasibility['candidate_parameters']
    need(baseline['loss_function']=='MAE' and baseline['eval_metric']=='MAE' and baseline['iterations']==2999
         and params==dict(baseline,loss_function='Huber:delta=1'), 'Synthetic passed-parameter comparison differs')
    for field in ('source_sha256','input_sha256'):
        for name,digest in feasibility[field].items():add(name,digest)
    need(set(method_sources)=={'source_sha256','evidence_sha256','official_urls'}
         and isinstance(method_sources['official_urls'],list) and method_sources['official_urls']
         and all(isinstance(url,str) and url.startswith('https://catboost.ai/') for url in method_sources['official_urls']),
         'Wrong source-only method inventory')
    for field in ('source_sha256','evidence_sha256'):
        need(isinstance(method_sources[field],dict) and method_sources[field], 'Missing source history')
        for name,digest in method_sources[field].items():add(name,digest)
    models=feasibility['models'];need(set(models)=={'huber_candidate','mae_control'}, 'Exactly two artificial models required')
    for name,item in models.items():
        add(item['model_path'],item['model_sha256'])
        need(item['model_path'].startswith('outputs/round22_research/') and item['model_path'].endswith('.cbm'),
             'Synthetic model outside research namespace')
        need(item['tree_count']==2 and item['feature_count']==95 and item['fit_rows']>0 and item['query_rows']>0
             and item['saved_reload_predictions_exact'] is True, 'Artificial model scope or reload check differs')
        need(len(item['categorical_positions'])==len(set(item['categorical_positions']))==23
             and all(type(k) is int and 0<=k<95 for k in item['categorical_positions']), 'Artificial category schema differs')
        passed=params if name=='huber_candidate' else baseline
        need(item['passed_parameters']==dict(passed,iterations=2,verbose=False), 'Artificial execution overrides differ')
        for phase in ('fitted_native_parameters','saved_native_parameters'):
            native=item[phase]
            if name=='huber_candidate':validate_objective_contract({key:native[key] for key in NATIVE_OBJECTIVE_CONTRACT})
            else:
                need(native['loss_function']=='MAE' and native['eval_metric']=='MAE'
                     and native['leaf_estimation_method']=='Exact' and type(native['leaf_estimation_iterations']) is int
                     and native['leaf_estimation_iterations']==1 and native['boost_from_average'] is True,
                     'Artificial control native defaults differ')
    a,b=(models[name]['saved_native_parameters'] for name in ('mae_control','huber_candidate'))
    need(set(a)==set(b), 'Synthetic saved native parameter inventories differ')
    difference={key:{'control':a[key],'candidate':b[key]} for key in a if a[key]!=b[key]}
    intended={'loss_function':{'control':'MAE','candidate':'Huber:delta=1'},
      **{key:{'control':value['from'],'candidate':value['to']} for key,value in IMPLICIT_NATIVE_POLICY_CHANGES.items()}}
    need(difference==feasibility['native_control_candidate_differences']==intended, 'Observed implicit defaults differ')
    need(research==expected, 'Frozen research closure omits named compatibility/history evidence')
    return {'frozen_research_dependency_count':len(research),'dependency_sha256':copy.deepcopy(research),
      'required_research_seed_paths':list(declared),'research_type':'synthetic_GPU_Huber_compatibility_and_source_history',
      'feasibility_path':FEASIBILITY_PATH,'feasibility_sha256':research[FEASIBILITY_PATH],
      'feasibility_status':feasibility['status'],'synthetic_only':True,'synthetic_model_count':2,
      'synthetic_tree_count_each':2,'declared_delta_TWD':feasibility['declared_delta_TWD'],
      'native_control_candidate_differences':copy.deepcopy(difference),
      'synthetic_model_artifacts':{name:{'path':item['model_path'],'sha256':item['model_sha256']} for name,item in models.items()},
      'limitations':copy.deepcopy(feasibility['limitations']),
      'method_sources_path':METHOD_SOURCES_PATH,'method_sources_sha256':research[METHOD_SOURCES_PATH],
      'official_urls':copy.deepcopy(method_sources['official_urls']),
      'report_recomputed_research_or_price_metrics':False}


def check_research(plan, bindings):
    name='src/round22_experiment.py'
    bindings.add(name,plan['source_sha256'][name])
    declared=research_expectations(safe(name).read_text(encoding='utf-8-sig'))
    need(len(declared)==27, 'Frozen R22 research must contain 27 identities')
    feasibility=bindings.authority(FEASIBILITY_PATH)
    methods=bindings.authority(METHOD_SOURCES_PATH)
    summary=validate_research(plan['research_dependency_sha256'],plan['dependency_sha256'],declared,feasibility,methods)
    bindings.mapping(declared)
    for field in ('source_sha256','input_sha256'):bindings.mapping(feasibility[field])
    for field in ('source_sha256','evidence_sha256'):bindings.mapping(methods[field])
    for item in feasibility['models'].values():bindings.add(item['model_path'],item['model_sha256'])
    return summary


def check_objective_evidence(model, proof, state, plan):
    for item in (model,state,plan):validate_objective_contract(item['native_objective_contract'])
    validate_objective_contract(proof['native_objective_contract'])
    need(proof['implicit_native_policy_changes']==plan['implicit_native_policy_changes']==IMPLICIT_NATIVE_POLICY_CHANGES,
         'Independent objective-policy changes differ')
    need(model['implicit_native_default_differences']==state['implicit_native_default_differences']
         ==plan['feature_rules']['implicit_native_default_differences']==IMPLICIT_NATIVE_DEFAULT_DIFFERENCES,
         'Helper implicit defaults evidence differs')
    validate_objective_contract(proof['independent_fit_state_evidence']['native_objective_contract'])
    content={k:v for k,v in state.items() if k!='state_content_sha256'}
    digest=hashlib.sha256(json.dumps(content,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
    need(digest==state['state_content_sha256']==model['feature_state_content_sha256'], 'Feature-state content identity differs')
    need(state['feature_rules']==plan['feature_rules'] and state['parameters']==model['parameters']
         and state['baseline_parameters']==model['baseline_parameters']
         and state['changed_parameter_names']==['loss_function'] and state['added_feature_names']==[]
         and state['removed_feature_names']==[] and state['baseline_feature_count']==95
         and state['original95_preserved'] is True, 'Saved feature-state contract differs')
    need(state['component']==plan['component'] and state['target_mode']=='raw'
         and state['target_formula']==plan['target_formula'] and state['inverse_formula']==plan['inverse_formula']
         and state['all_feature_names']==state['geo_state']['all_feature_names']
         and len(state['all_feature_names'])==len(set(state['all_feature_names']))==95
         and len(state['categorical_columns'])==len(set(state['categorical_columns']))==23,
         'Saved original95 feature-state schema differs')
    need(state['source_sha256']==model['component_source_sha256']
         and state['fit_model_feature_values_sha256']==model['fit_model_feature_values_sha256']
         and state['fit_rows']==model['fit_rows'] and model['raw_z_units']=='million_TWD'
         and all(state[key] is False for key in ('fit_state_uses_y','fit_state_uses_query','query_labels_used')),
         'Saved state sources/data-use evidence differs')


def validate_artifact_qa(qa, expected):
    need(qa['round']==22 and qa['status']=='passed' and qa['checked_stages']==['development']
         and qa['protected_unchanged'] is True and qa['protected_before']==qa['protected_after'], 'Report QA incomplete or protected bytes changed')
    for key,value in expected.items():need(qa[key]==value, 'Report QA identity differs: '+key)
    need(type(qa['check_count']) is int and qa['check_count']>0 and len(qa['checks'])==qa['check_count'],
         'Report QA check inventory differs')


def check_artifact_qa(plan_digests, b):
    qa=b.authority(QA_PATH)
    source='src/round22_report_qa.py';b.add(source,qa['script_sha256'])
    expected={'script_path':source,'script_sha256':b.values[source]['sha256'],
      'report_source_sha256':b.values['src/round22_report.py']['sha256'],
      'report_sha256':sha(safe('reports/round22_report.md')),'report_results_sha256':plan_digests['report'],
      'plan_sha256':plan_digests['plan'],'frozen_candidate_sha256':plan_digests['frozen'],
      'accepted_candidate_sha256':plan_digests['accepted']}
    validate_artifact_qa(qa,expected)
    return b.values[QA_PATH]['sha256']


def collect():
    """Read finalized authorities and hashes. Never inspect npy/model contents."""
    files = inventory()
    b = Bindings()
    work_log_name = 'reports/round22_work_log.md'
    work_log_bytes = safe(work_log_name).read_bytes()
    b.add(work_log_name, closed_work_log_sha256(work_log_bytes))
    authority_paths = {
        'plan': 'outputs/round22/candidate_plan.json',
        'frozen': 'outputs/round22/frozen_candidate.json',
        'accepted': 'outputs/round22/accepted_candidate.json',
        'result': 'outputs/round22/evaluation/development/results.json',
        'model': 'outputs/round22/models/development/metadata.json',
        'proof': 'outputs/round22/audit/development_replay.json',
        'report': 'reports/round22_results.json',
    }
    values = {key: b.authority(name) for key, name in authority_paths.items()}
    plan, frozen, accepted, result, model, proof, report = (values[k] for k in authority_paths)
    digests = {key: b.values[name]['sha256'] for key, name in authority_paths.items()}
    rejection_contract(plan, frozen, accepted, result)
    for value in (frozen, accepted, result, model, proof):
        need(value['plan_sha256'] == digests['plan'], 'Authority refers to another R22 plan')
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
    b.add('src/round22_report.py', report['source_input_sha256']['src/round22_report.py'])
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
    expected_outputs = {'outputs/round22/evaluation/development/' + name + '.npy'
                        for name in (CONTROL, CANDIDATE)}
    need(set(result['output_artifact_sha256']) == expected_outputs, 'Unexpected prediction inventory')
    need(model['round'] == 22 and model['stage'] == 'development'
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
         == plan['changed_parameter_names'] == CHANGED_PARAMETERS
         and model['removed_feature_names'] == proof['removed_feature_names'] == plan['removed_feature_names'] == REMOVED_FEATURES,
         'Fixed parameter or original-feature evidence differs')
    need(model['added_feature_names']==proof['added_feature_names']==plan['added_feature_names']==ADDED_FEATURES
         and model['baseline_feature_count']==95 and model['original95_preserved'] is True, 'Original95 preservation differs')
    validate_parameter_change(model['baseline_parameters'],model['parameters'])
    check_capacity(model, proof)
    feature_state=read(safe('outputs/round22/models/development/feature_state.json'))
    check_objective_evidence(model,proof,feature_state,plan)
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
    need(proof['round'] == 22 and proof['stage'] == 'development' and proof['status'] == 'passed'
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
    need(state_evidence['added_feature_names']==[], 'Independent feature-state addition differs')
    validate_objective_contract(state_evidence['native_objective_contract'])
    b.add('src/round22_verify_backtest.py', proof['source_sha256'])
    check_report(report, plan, accepted, result, model, proof, digests['proof'])
    report_qa_sha=check_artifact_qa(digests,b)
    # These three source maps are authoritative; verify that the independent
    # replay actually bound the artifacts it says it replayed, without loading them.
    required_proof = {authority_paths[k] for k in ('plan', 'frozen', 'accepted', 'result', 'model')}
    required_proof.update('outputs/round22/models/development/' + name
                          for name in MODEL_FILES)
    required_proof.update(expected_outputs)
    need(required_proof <= set(proof['source_input_sha256']), 'Replay input closure is incomplete')
    required_report = set(required_proof) | {authority_paths['accepted'], authority_paths['proof'],
                                           'src/round22_report.py'}
    need(required_report <= set(report['source_input_sha256']), 'Report input closure is incomplete')
    # Attempts are copied, not interpreted as current source bindings.
    for name in sorted(files):
        b.add(name)
        if name.endswith('.sha256'):
            path = safe(name)
            target = path.with_suffix('.json')
            need(target.is_file(), 'Historical/current receipt has no JSON: ' + name)
            locked(target)  # Byte receipt only; do not traverse historical source maps.
    need(inventory() == files, 'R22 inventory changed during collection')
    need(all(not copy_scope(name) or name in files for name in b.values), 'Owned dependency omitted from copy inventory')
    copied = {name: b.values[name].copy() for name in sorted(files)}
    external = {name: item.copy() for name, item in sorted(b.values.items()) if name not in copied}
    need(not MUTABLE.intersection(set(copied) | set(external)), 'Mutable entry point became permanent')
    registry = safe('runs/versions.json')
    context = {'path': 'runs/versions.json', 'sha256': sha(registry), 'bytes': registry.stat().st_size,
               'plan_time_sha256': plan['registry_sha256_at_plan'],
               'authority_observed_sha256': sorted(b.mutable_declarations.get('runs/versions.json', set())),
               'verification_policy': 'creation-time observation only; never a future external guard'}
    state = {'round': 22, 'is_release': False, 'accepted_backtest': False,
             'candidate': CANDIDATE, 'selected_candidate': CONTROL,
             'stopping_reason': accepted['stopping_reason'], 'stage_status': accepted['stage_status'],
             'recommended_version': 'v3', 'recommended_submission': 'outputs/submission_v3.csv',
             'new_submission_created': False, 'registry_modified': False,
             'plan_sha256': digests['plan'], 'frozen_candidate_sha256': digests['frozen'],
             'accepted_candidate_sha256': digests['accepted'], 'development_result_sha256': digests['result'],
             'independent_replay_sha256': digests['proof'], 'report_json_sha256': digests['report'],
             'source_path': 'src/archive_round22.py', 'source_sha256': sha(__file__),
             'property_labels_or_prediction_values_read_by_archiver': False,
             'array_or_model_deserialization_performed': False, 'models_refitted_or_MAE_recomputed': False,
             'index_identity_uses_saved_contract_and_replay_not_array_deserialization': True,
             'historically_untouched': False, 'historical_attempts_kept_without_current_source_enforcement': True,
             'registry_context': context, 'current_status_is_not_bound': True,
             'research_dependency_sha256': plan['research_dependency_sha256'],
             'research_is_artificial_compatibility_and_source_history': True,
             'synthetic_CBM_files_are_not_formal_price_models': True,
             'report_artifact_qa_sha256':report_qa_sha,
             'new_research_folder_and_receipts_preserved': True, 'research_summary': research_summary}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round22_development_rejection_experiments', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 22 and state['is_release'] is False and state['accepted_backtest'] is False
         and state['selected_candidate'] == CONTROL and state['stopping_reason'] == 'development_price_gates_failed'
         and state['recommended_version'] == 'v3' and state['new_submission_created'] is False,
         'Snapshot does not describe the finalized development rejection')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copied and external paths overlap')
    names = set(copied) | set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names} & {n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, RESEARCH_FILES,{QA_PATH,QA_PATH[:-5]+'.sha256'}) <= set(copied) and all(copy_scope(n) for n in copied),
         'Wrong R22 copy inventory')
    need(all(not copy_scope(n) for n in external), 'Owned R22 artifact left external')
    need(manifest['copied_file_count'] == len(copied)
         and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(item['bytes'] for item in copied.values()),
         'Manifest count/size mismatch')
    for name, item in {**copied, **external}.items():
        need(type(item['bytes']) is int and item['bytes'] >= 0
             and re.fullmatch('[0-9a-f]{64}', item['sha256']), 'Invalid file identity: ' + name)
    for key, name in [('plan_sha256', 'outputs/round22/candidate_plan.json'),
                      ('frozen_candidate_sha256', 'outputs/round22/frozen_candidate.json'),
                      ('accepted_candidate_sha256', 'outputs/round22/accepted_candidate.json'),
                      ('development_result_sha256', 'outputs/round22/evaluation/development/results.json'),
                      ('independent_replay_sha256', 'outputs/round22/audit/development_replay.json'),
                      ('report_json_sha256', 'reports/round22_results.json'),
                      ('report_artifact_qa_sha256', QA_PATH),
                      ('source_sha256', 'src/archive_round22.py')]:
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
        need(inventory() == set(copied), 'Original R22 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}


def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round22_experiments_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round22_experiments_staging_' + uuid.uuid4().hex)
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
    manifest = {'kind': 'round22_development_rejection_experiments',
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
    """Complete bounded archive safety, objective, research and saved-schema checks."""
    checks=work_log_self_check()
    def check(name,value):need(bool(value),'Synthetic check failed: '+name);checks.append(name)
    def reject(name,fn):
        try:fn()
        except (RuntimeError,KeyError,TypeError,ValueError,SyntaxError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    ok=check;rejects=reject
    baseline={'iterations':2999,'learning_rate':.065,'depth':8,'boosting_type':'Plain',
              'loss_function':'MAE','eval_metric':'MAE','random_seed':42,'thread_count':4,'l2_leaf_reg':6}
    params=dict(baseline,loss_function='Huber:delta=1')
    validate_parameter_change(baseline,params);checks.append('only_passed_Huber_loss_changes')
    for key,value in [('iterations',5998),('learning_rate',.0325),('depth',6),('boosting_type','Ordered'),
                      ('loss_function','MAE'),('loss_function','Huber:delta=2'),('eval_metric','RMSE')]:
        wrong={**params,key:value};reject('parameter_'+key+'_'+str(value),lambda p=wrong:validate_parameter_change(baseline,p))
    for key in ('leaf_estimation_method','leaf_estimation_iterations','boost_from_average'):
        bb=dict(baseline,**{key:NATIVE_OBJECTIVE_CONTRACT[key]});pp=dict(bb,loss_function='Huber:delta=1')
        reject('implicit_default_not_passed_'+key,lambda b=bb,p=pp:validate_parameter_change(b,p))
    rules={'feature_count':95,'categorical_feature_count':23,'baseline_feature_count':95,'original95_preserved':True,
      'existing_95_values_dtypes_order_and_index_unchanged':True,'removed_feature_names':[],'added_feature_names':[],
      'changed_parameter_names':['loss_function'],'native_objective_contract':copy.deepcopy(NATIVE_OBJECTIVE_CONTRACT),
      'baseline_loss_function':'MAE','loss_function':'Huber:delta=1','eval_metric':'MAE','huber_delta_in_model_units':1.,
      'huber_delta_TWD':1e6,'raw_z_units':'million_TWD','implicit_native_default_differences':copy.deepcopy(IMPLICIT_NATIVE_DEFAULT_DIFFERENCES),
      'native_defaults_explicitly_passed':False,'same_passed_parameters_does_not_imply_same_native_defaults':True}
    validate_huber_rules(rules);checks.append('original95_Huber_rules')
    for key,value in [('feature_count',97),('categorical_feature_count',25),('added_feature_names',['grid']),
                      ('huber_delta_TWD',2e6),('native_defaults_explicitly_passed',True),('original95_preserved',False)]:
        wrong=copy.deepcopy(rules);wrong[key]=value
        reject('rules_'+key,lambda r=wrong:validate_huber_rules(r))
    model={'actual_model_depth':8,'actual_grow_policy':'SymmetricTree','actual_trees':2999,
      'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,'actual_boosting_type':'Plain',
      'actual_learning_rate':.06499999762,'fitted_actual_learning_rate':.06499999761581421,
      'native_categorical_feature_count':23,'native_learning_rate_comparison_rtol':1e-7,
      'native_learning_rate_comparison_atol':0.,**{'actual_'+k:v for k,v in NATIVE_OBJECTIVE_CONTRACT.items()}}
    proof={k:model[k] for k in ('native_categorical_feature_count','native_learning_rate_comparison_rtol','native_learning_rate_comparison_atol')}
    proof['native_model_capacity']={k:model[k] for k in CAPACITY_FIELDS}
    check_capacity(model,proof);checks.append('native12_saved_fitted_LR_pair_and23cats')
    for key,value in [('actual_trees',5998),('native_categorical_feature_count',25),('actual_learning_rate',.0325),
                      ('actual_learning_rate',float('nan')),('maximum_tree_leaf_count',257),('actual_loss_function','MAE'),
                      ('actual_loss_function','Huber:delta=2'),('actual_eval_metric','RMSE'),('actual_leaf_estimation_method','Exact'),
                      ('actual_leaf_estimation_iterations',2),('actual_leaf_estimation_iterations',True),
                      ('actual_boost_from_average',True),('actual_boost_from_average',0)]:
        wrong={**model,key:value};reject('native_'+key+'_'+str(value),lambda m=wrong:check_capacity(m,proof))
    plan={'round':22,'candidate':CANDIDATE,'control':CONTROL,'historically_untouched':False,
      'feature_count':95,'categorical_count':23,'parameters':copy.deepcopy(params),'baseline_parameters':copy.deepcopy(baseline),
      'component':'geography_huber_catboost','sample_weight':'uniform','target_mode':'raw','weight_argument_passed_to_pool':False,
      'changed_parameter_names':['loss_function'],'model_hyperparameter_change':{'loss_function':{'from':'MAE','to':'Huber:delta=1'}},
      'capacity_constraint':{'depth':8,'max_leaves_per_tree':256,'grow_policy':'SymmetricTree','boosting_type':'Plain'},
      'native_learning_rate_contract':dict(LR_CONTRACT),'native_objective_contract':dict(NATIVE_OBJECTIVE_CONTRACT),
      'implicit_native_policy_changes':copy.deepcopy(IMPLICIT_NATIVE_POLICY_CHANGES),'removed_feature_names':[],'added_feature_names':[],
      'feature_rules':copy.deepcopy(rules),'target_formula':'y_TWD / 1e6','inverse_formula':'max(z * 1e6, 0)',
      'split_contract':{stage:{'fit_rows':7,'query_rows':4} for stage in STAGES},
      'splits':{stage:['start','end'] for stage in STAGES},'formulas':{'fixed':'synthetic'},'thresholds':{'fixed':'synthetic'},'target_group_weights':{}}
    checks_dict={'overall_improves_at_least_1pct':False,'weighted_improves_at_least_1pct':False,
      'quarters_deteriorate_at_most_5pct':True,'large_strata_deteriorate_at_most_5pct':True}
    result={'round':22,'stage':'development','evaluated_candidates':[CONTROL,CANDIDATE],
      'decision':{'passed':False,'selected_candidate':CONTROL,'checks':checks_dict},'historically_untouched':False,
      'fit_rows':7,'query_rows':4,'formulas':plan['formulas'],'thresholds':plan['thresholds'],'target_group_weights':{},
      'metrics':{'synthetic_saved_summary':'opaque; no recomputation'}}
    frozen={'round':22,'candidate':CANDIDATE,'passed':False,'later_stages_allowed':False,'selected_candidate':CONTROL,
      'historically_untouched':False,'formulas':plan['formulas'],'thresholds':plan['thresholds'],'parameters':params,'target_group_weights':{}}
    accepted={'round':22,'candidate':CANDIDATE,'accepted':False,'selected_candidate':CONTROL,
      'stopping_reason':'development_price_gates_failed','deployment_decision':'keep_v3',
      'stage_status':{stage:'price_evaluated' if stage=='development' else 'not_executed_after_development_rejection' for stage in STAGES},
      'unevaluated_stages':list(STAGES[1:]),'stage_result_sha256':{'development':'a'*64},'new_submission_created':False,
      'currently_recommended_submission':'outputs/submission_v3.csv','no_reselection_or_parameter_change':True,'historically_untouched':False}
    rejection_contract(plan,frozen,accepted,result);checks.append('finalized_development_rejection_contract')
    for key,value in [('accepted',True),('selected_candidate',CANDIDATE),('new_submission_created',True),
                      ('unevaluated_stages',[]),('stopping_reason','later_backtest_gates_failed')]:
        wrong=copy.deepcopy(accepted);wrong[key]=value
        reject('accepted_'+key,lambda a=wrong:rejection_contract(plan,frozen,a,result))
    wrong=copy.deepcopy(frozen);wrong['later_stages_allowed']=True
    reject('later_authorization_not_dev_rejection',lambda:rejection_contract(plan,wrong,accepted,result))
    wrong=copy.deepcopy(result);wrong['decision']['checks']['overall_improves_at_least_1pct']=True
    wrong['decision']['checks']['weighted_improves_at_least_1pct']=True
    reject('saved_gate_boolean_inconsistency',lambda:rejection_contract(plan,frozen,accepted,wrong))
    wrong=copy.deepcopy(plan);wrong['weight_argument_passed_to_pool']=True
    reject('plan_Pool_weight_rejected',lambda:rejection_contract(wrong,frozen,accepted,result))
    for item in (model,proof):item['native_objective_contract']=dict(NATIVE_OBJECTIVE_CONTRACT)
    proof['implicit_native_policy_changes']=copy.deepcopy(IMPLICIT_NATIVE_POLICY_CHANGES)
    proof['independent_fit_state_evidence']={'native_objective_contract':dict(NATIVE_OBJECTIVE_CONTRACT)}
    state={'feature_rules':copy.deepcopy(rules),'parameters':params,'baseline_parameters':baseline,
      'changed_parameter_names':['loss_function'],'added_feature_names':[],'removed_feature_names':[],'baseline_feature_count':95,
      'original95_preserved':True,'native_objective_contract':dict(NATIVE_OBJECTIVE_CONTRACT),
      'implicit_native_default_differences':copy.deepcopy(IMPLICIT_NATIVE_DEFAULT_DIFFERENCES),
      'component':plan['component'],'target_mode':'raw','target_formula':plan['target_formula'],'inverse_formula':plan['inverse_formula'],
      'all_feature_names':['f'+str(i) for i in range(95)],'categorical_columns':['f'+str(i) for i in range(23)],
      'geo_state':{'all_feature_names':['f'+str(i) for i in range(95)]},'source_sha256':{'synthetic.py':'a'*64},
      'fit_model_feature_values_sha256':'b'*64,'fit_rows':7,'fit_state_uses_y':False,'fit_state_uses_query':False,'query_labels_used':False}
    state['state_content_sha256']=hashlib.sha256(json.dumps(state,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()
    model.update(parameters=params,baseline_parameters=baseline,feature_state_content_sha256=state['state_content_sha256'],
      implicit_native_default_differences=copy.deepcopy(IMPLICIT_NATIVE_DEFAULT_DIFFERENCES),component_source_sha256=state['source_sha256'],
      fit_model_feature_values_sha256='b'*64,fit_rows=7,raw_z_units='million_TWD')
    check_objective_evidence(model,proof,state,plan);checks.append('state_meta_plan_proof_objective_source_identity')
    wrong=copy.deepcopy(state);wrong['fit_state_uses_query']=True
    reject('mutated_saved_state_rejected',lambda:check_objective_evidence(model,proof,wrong,plan))
    wrong=copy.deepcopy(proof);wrong['native_objective_contract']['boost_from_average']=True
    reject('proof_objective_contradiction',lambda:check_objective_evidence(model,wrong,state,plan))
    source_name='src/synthetic_recipe.py';method_name='reports/synthetic_method.md';config_name='synthetic/config.json'
    research={FEASIBILITY_PATH:'a'*64,FEASIBILITY_PATH[:-5]+'.sha256':'b'*64,
              METHOD_SOURCES_PATH:'c'*64,METHOD_SOURCES_PATH[:-5]+'.sha256':'d'*64,
              source_name:'e'*64,method_name:'f'*64,config_name:'1'*64,
              'outputs/round22_research/mae_control.cbm':'2'*64,'outputs/round22_research/huber_candidate.cbm':'3'*64}
    control_native={'loss_function':'MAE','eval_metric':'MAE','leaf_estimation_method':'Exact',
        'leaf_estimation_iterations':1,'boost_from_average':True,'iterations':2,'learning_rate':.06499999762}
    huber_native=dict(control_native,**NATIVE_OBJECTIVE_CONTRACT)
    models={}
    for name,native,passed,digest in [('mae_control',control_native,baseline,'2'*64),('huber_candidate',huber_native,params,'3'*64)]:
        models[name]={'model_path':'outputs/round22_research/'+name+'.cbm','model_sha256':digest,'tree_count':2,
            'feature_count':95,'fit_rows':8,'query_rows':2,'saved_reload_predictions_exact':True,
            'categorical_positions':list(range(21))+[93,94],'passed_parameters':dict(passed,iterations=2,verbose=False),
            'fitted_native_parameters':copy.deepcopy(native),'saved_native_parameters':copy.deepcopy(native)}
    observed={key:{'control':control_native[key],'candidate':huber_native[key]} for key in control_native if control_native[key]!=huber_native[key]}
    feasibility={'round':22,'status':'synthetic_GPU_compatibility_passed','synthetic_only':True,'synthetic_model_count':2,
        'property_MAE_computed':False,'property_X_y_models_or_predictions_loaded':False,
        'declared_candidate_loss':'Huber:delta=1','declared_delta_TWD':1000000,'candidate_passed_parameter_changes':['loss_function'],
        'feasibility_overrides':{'iterations':2,'verbose':False},'baseline_parameters':copy.deepcopy(baseline),
        'candidate_parameters':copy.deepcopy(params),'source_sha256':{source_name:'e'*64},'input_sha256':{config_name:'1'*64},
        'models':models,'native_control_candidate_differences':observed,'limitations':['Artificial compatibility only.']}
    methods={'source_sha256':{source_name:'e'*64},'evidence_sha256':{method_name:'f'*64},
             'official_urls':['https://catboost.ai/docs/en/concepts/loss-functions-regression']}
    source='RESEARCH_FILES = '+repr(tuple(research))+'\nRESEARCH_EXPECTED_SHA = '+repr(research)
    expected=research_expectations(source)
    summary=validate_research(research,research,expected,feasibility,methods)
    ok('synthetic_models_and_source_history_research_closure',summary['frozen_research_dependency_count']==len(research)
       and summary['synthetic_model_count']==2 and not summary['report_recomputed_research_or_price_metrics'])
    rejects('missing_research_dependency',lambda:validate_research(research,{},expected,feasibility,methods))
    bad=copy.deepcopy(expected);bad[source_name]='9'*64
    rejects('driver_map_identity_mismatch',lambda:validate_research(research,research,bad,feasibility,methods))
    for field,value in [('status','pending'),('synthetic_only',False),('synthetic_model_count',1),
                        ('property_MAE_computed',True),('property_X_y_models_or_predictions_loaded',True),
                        ('declared_delta_TWD',2e6),('candidate_passed_parameter_changes',[])]:
        bad=copy.deepcopy(feasibility);bad[field]=value
        rejects('research_'+field,lambda f=bad:validate_research(research,research,expected,f,methods))
    bad=copy.deepcopy(feasibility);bad['source_sha256']['unfrozen.py']='9'*64
    rejects('unfrozen_synthetic_source',lambda:validate_research(research,research,expected,bad,methods))
    bad=copy.deepcopy(feasibility);bad['models']['huber_candidate']['model_sha256']='9'*64
    rejects('synthetic_model_hash_conflict',lambda:validate_research(research,research,expected,bad,methods))
    for field,value in [('tree_count',2999),('saved_reload_predictions_exact',False),('categorical_positions',[0,0])]:
        bad=copy.deepcopy(feasibility);bad['models']['huber_candidate'][field]=value
        rejects('synthetic_model_'+field,lambda f=bad:validate_research(research,research,expected,f,methods))
    bad=copy.deepcopy(feasibility);bad['models']['huber_candidate']['saved_native_parameters']['boost_from_average']=True
    rejects('synthetic_policy_contradiction',lambda:validate_research(research,research,expected,bad,methods))
    bad=copy.deepcopy(methods);bad['evidence_sha256']['unfrozen.md']='9'*64
    rejects('unfrozen_method_evidence',lambda:validate_research(research,research,expected,feasibility,bad))
    bad=copy.deepcopy(methods);bad['status']='completed_pure_X_only'
    rejects('wrong_inherited_inventory_schema',lambda:validate_research(research,research,expected,feasibility,bad))
    for path in (FEASIBILITY_PATH,METHOD_SOURCES_PATH):
        receipt=path[:-5]+'.sha256';bad=copy.deepcopy(research);bad.pop(receipt)
        rejects('missing_receipt_'+receipt,lambda m=bad:validate_research(m,m,m,feasibility,methods))
    rejects('missing_research_declaration',lambda:research_expectations('OTHER = {}'))
    rejects('dynamic_declaration_not_executed',lambda:research_expectations('RESEARCH_FILES=call()\nRESEARCH_EXPECTED_SHA={}'))
    # Report fields are compared as saved data, without metric calculation.
    complete_model={key:model.get(key,'synthetic') for key in MODEL_SUMMARY_FIELDS}
    report_value={'round':22,'candidate':CANDIDATE,'accepted':False,'currently_recommended_submission':'outputs/submission_v3.csv',
      'stopping_reason':accepted['stopping_reason'],'deployment_decision':'keep_v3','new_submission_created':False,
      'historically_untouched':False,'public_score_used':False,'private_score_known':False,
      'report_only_reads_saved_JSON_and_hashes_binary_inputs':True,'stages':[]}
    proof.update(replays={},saved_raw_z_bitwise_equal=True,removed_feature_names=[],added_feature_names=[])
    for stage in STAGES:
        row={'stage':stage,'status':accepted['stage_status'][stage],'dates':plan['splits'][stage],
             'fit_rows':7,'query_rows':4,'metrics':None,'decision':None,'model':None,'independent_review':None}
        if stage=='development':row.update(metrics=result['metrics'],decision=result['decision'],model=copy.deepcopy(complete_model),
            independent_review={'proof_path':'outputs/round22/audit/development_replay.json','proof_sha256':'c'*64,
                'replays':{},'raw_z_bitwise_equal':True,'native_model_capacity':proof['native_model_capacity'],
                'removed_feature_names':[],'added_feature_names':[],
                'native_objective_contract':dict(NATIVE_OBJECTIVE_CONTRACT),
                'implicit_native_policy_changes':copy.deepcopy(IMPLICIT_NATIVE_POLICY_CHANGES),
                'independent_fit_state_evidence':copy.deepcopy(proof['independent_fit_state_evidence'])})
        report_value['stages'].append(row)
    check_report(report_value,plan,accepted,result,complete_model,proof,'c'*64);checks.append('report_saved_native12_and_null_later_schema')
    wrong=copy.deepcopy(report_value);wrong['stages'][1]['metrics']={}
    reject('report_fabricated_later_results',lambda:check_report(wrong,plan,accepted,result,complete_model,proof,'c'*64))
    wrong=copy.deepcopy(report_value);wrong['stages'][0]['independent_review']['native_objective_contract']={}
    reject('report_omitted_objective_proof',lambda:check_report(wrong,plan,accepted,result,complete_model,proof,'c'*64))
    qa_expected={'script_path':'src/round22_report_qa.py','script_sha256':'a'*64,'report_source_sha256':'b'*64,
      'report_sha256':'c'*64,'report_results_sha256':'d'*64,'plan_sha256':'e'*64,'frozen_candidate_sha256':'f'*64,
      'accepted_candidate_sha256':'1'*64}
    qa={'round':22,'status':'passed','checked_stages':['development'],'protected_unchanged':True,
      'protected_before':{'v3':'a'*64},'protected_after':{'v3':'a'*64},'checks':['a'],'check_count':1,**qa_expected}
    validate_artifact_qa(qa,qa_expected);checks.append('artifact_QA_source_report_plan_protected_binding')
    for key,value in [('protected_after',{}),('checked_stages',['development','confirmation']),
                      ('report_results_sha256','f'*64),('status','pending')]:
        wrong=copy.deepcopy(qa);wrong[key]=value
        reject('artifact_QA_'+key,lambda q=wrong:validate_artifact_qa(q,qa_expected))
    validate_current_stage_names({'development'});checks.append('development_only_inventory')
    reject('later_stage_rejected',lambda:validate_current_stage_names({'development','confirmation'}))
    for name in ('models/.development-any/model.cbm','evaluation/.stage-1/a.npy','foo.part'):
        reject('current_staging_'+name,lambda n=name:validate_current_paths({n}))
    check('seven_owned_sources_and_entire_research_copyable',len(SOURCES)==7 and all(copy_scope(n) for n in SOURCES+DOCS+RESEARCH_FILES)
      and copy_scope(QA_PATH) and copy_scope('outputs/round22_research/huber_feasibility.log'))
    check('runner_logs_preserved',copy_scope('outputs/round22/pipeline_runner.py') and copy_scope('outputs/round22/logs/report.log'))
    check('old_maps_mutables_not_copied',not any(copy_scope(n) for n in ('runs/versions.json','CURRENT_STATUS.md','src/round6_geo.py',
      'outputs/round6/geography/final/train.pkl','outputs/round20_pre_score_failure_01/a.json')))
    for name in ('../escape','C:/escape','a//b','a/./b','a/../b','a\x00b'):
        reject('unsafe_path_'+repr(name),lambda n=name:normalized(n))
    binding=Bindings();binding.declare('src/a.py','a'*64);binding.values['src/a.py']={'sha256':'a'*64,'bytes':0}
    reject('conflicting_authority_rejected',lambda:binding.declare('src/a.py','b'*64))
    reject('case_ambiguous_source_rejected',lambda:binding.declare('SRC/a.py','a'*64))
    check('registry_context_only',binding.declare('runs/versions.json','c'*64) is False and 'runs/versions.json' not in binding.values)
    check('native12_and_matching_summary_schema',len(CAPACITY_FIELDS)==12 and set(CAPACITY_FIELDS)<=set(MODEL_SUMMARY_FIELDS)
      and {'native_objective_contract','implicit_native_default_differences','raw_z_units'}<=set(MODEL_SUMMARY_FIELDS))
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

