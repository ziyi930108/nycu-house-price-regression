"""R24: one fixed equal-weight three-seed geography CatBoost ensemble of v3.
Only the three declared backtests are implemented. No final fit or CSV release.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor, Pool
import round6_common as c6
import round6_evaluate as e6
import round24_seed_ensemble as component
from round6_prepare_geo import load_geographic_partition
from round6_geo import frame_sha
from round2_neighborhood import predict_reference
from round3_evaluate import score, target_weights, runtime_versions

ROOT = c6.ROOT
OUT = ROOT / 'outputs/round24'
STAGES = tuple(c6.SPLITS)
STAGE_YEAR = {'development': 2022, 'confirmation': 2023, 'auxiliary': 2021, 'final': 2024}
CANDIDATE = 'seed_ensemble_geography_blend'
CONTROL = 'v3_control'
EXTRA_SOURCES = ()
REQUIRED_DOCUMENTS = (
    'reports/round24_plan.md',
    'reports/round24_method_review.md',
    'reports/round24_pipeline_review.md',
)
REMOVED_FEATURES = []
ADDED_FEATURES = []
# Frozen pure-X research recipe, result, provenance, and review identities.
RESEARCH_FILES = ('outputs/round24_research/method_sources.json', 'outputs/round24_research/method_sources.sha256')
HELPER_SHA = '45bb6c7d2ef881516d3c9ec57fa36c29618d013091d3eb753168c73748794b2c'
SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA = '52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA = 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
THRESHOLDS = dict(e6.THRESHOLDS)
FORMULAS = {
    CONTROL: '.375*old_raw_cat+.375*geo_raw_cat+.25*old_unit_lgb; then v2 trend once',
    CANDIDATE: '.375*old_raw_cat+.375*mean_of_three_seed_clipped_TWD+.25*old_unit_lgb; then v2 trend once',
}
BRANCH_FILES = {'model.cbm','feature_state.json','raw_z.npy','predictions.npy','fit_indices.npy','query_indices.npy','metadata.json','metadata.sha256'}
MODEL_FILES = {'predictions.npy','fit_indices.npy','query_indices.npy','seed42_raw_z.npy','seed42_predictions.npy',
               'seed42_replay.json','seed42_replay.sha256'} | {f'seed_{seed}/{name}' for seed in component.NEW_SEEDS for name in BRANCH_FILES}
NATIVE_CAPACITY_FIELDS = ('actual_model_depth','actual_boosting_type','actual_grow_policy','actual_trees',
                         'maximum_tree_leaf_count','tree_leaf_counts_sha256','actual_learning_rate','actual_random_seed')
require, read, write, locked, locked_write, sha, rel = (
    c6.require, c6.read, c6.write, c6.read_locked, c6.locked_write, c6.sha256, c6.rel)

def now():return datetime.now(timezone.utc).isoformat()

def value_sha(a,dtype):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()

def sources():
    pending=[Path(__file__).resolve()]+[ROOT/name for name in EXTRA_SOURCES];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/'src'/(name.split('.')[0]+'.py')
                if local.is_file() and local not in found:pending.append(local)
    return {rel(p):sha(p) for p in sorted(found)}

def stage_paths(stage):
    base=c6.baseline_folder(stage)
    geo=c6.OUT/'models'/stage/'geography_catboost'
    evaluation=c6.OUT/'evaluation'/stage
    paths=[base/n for n in ['catboost_raw.npy','lightgbm_l1_unit63.npy','reference_v2.pkl','fit_indices.npy','check_indices.npy']]
    paths += [geo/n for n in ['model.cbm','feature_state.json','metadata.json','predictions.npy','fit_indices.npy','query_indices.npy']]
    paths += [evaluation/n for n in ['results.json','results.sha256','geography_blend.npy']]
    manifest=c6.OUT/'geography'/stage/'manifest.json'
    paths += [manifest,manifest.with_suffix('.sha256')]
    item=locked(manifest)
    paths += [ROOT/v['path'] for v in item['features'].values()]
    paths += [ROOT/p for p in item['index_files_sha256']]
    paths += [ROOT/p for p in item['source_sha256']]
    return sorted(set(paths))

def load_model(path):
    model=CatBoostRegressor()
    model.load_model(str(path))
    return model

def check_later_gate(stage,frozen,plan_sha):
    require(stage in ('confirmation','auxiliary'),'Only declared later backtests may be authorized')
    require(frozen['passed'] is True and frozen['later_stages_allowed'] is True and frozen['selected_candidate']==CANDIDATE,'Development rejected; later stages prohibited')
    require(frozen['plan_sha256']==plan_sha,'Development selection belongs to another plan')

def partition(stage):
    data,X,y,Xq,yq,fi,qi=c6.partition(stage)
    spec=locked(OUT/'candidate_plan.json')['split_contract'][stage]
    for label,array in [('fit',fi),('query',qi)]:
        require(spec[label+'_indices_content_sha256']==value_sha(array,'<i8'),'R24 partition index differs')
    base=c6.baseline_folder(stage);old=c6.OUT/'models'/stage/'geography_catboost'
    e6.verify_indices(base,fi,qi,'check_indices.npy');e6.verify_indices(old,fi,qi,'query_indices.npy')
    return data,X,y,Xq,yq,fi,qi

def publish(staging,destination):
    require(staging.resolve().parent==destination.resolve().parent and staging.resolve().is_relative_to(OUT.resolve()),'Unsafe R24 publication path')
    require(not destination.exists(),'Refusing to replace recorded experiment')
    staging.rename(destination)

def blend(old,geo,unit,reference,Xq,apply_reference=predict_reference):
    base=.75*(.5*np.asarray(old)+.5*np.asarray(geo))+.25*np.asarray(unit)
    prediction=np.asarray(apply_reference(reference,Xq,base,variant='trend'),dtype='float64')
    require(prediction.shape==(len(Xq),) and np.isfinite(prediction).all() and (prediction>=0).all(),'Invalid blended TWD predictions')
    return prediction

def decide(stage,metrics):
    require(set(metrics)=={CONTROL,CANDIDATE},'Only the fixed control and candidate may be scored')
    base,value=metrics[CONTROL],metrics[CANDIDATE]
    large=e6.large_strata_ok(value,base)
    improved=[q for q,item in base['forecast_quarters'].items() if item['rows'] and value['forecast_quarters'][q]['mae']<item['mae']]
    if stage=='development':
        checks={'overall_improves_at_least_1pct':value['overall_mae']<=(1-THRESHOLDS['development_min_overall_improvement'])*base['overall_mae'],'weighted_improves_at_least_1pct':value['weighted_mae']<=(1-THRESHOLDS['development_min_weighted_improvement'])*base['weighted_mae'],'quarters_deteriorate_at_most_5pct':all(value['forecast_quarters'][q]['mae']<=(1+THRESHOLDS['development_max_quarter_deterioration'])*item['mae'] for q,item in base['forecast_quarters'].items() if item['rows']),'large_strata_deteriorate_at_most_5pct':large}
    else:
        require(stage in ('confirmation','auxiliary'),'Unknown evaluation stage')
        checks={'overall_within_limit':value['overall_mae']<=(1+THRESHOLDS[stage+'_max_overall_deterioration'])*base['overall_mae'],'weighted_within_limit':value['weighted_mae']<=(1+THRESHOLDS[stage+'_max_weighted_deterioration'])*base['weighted_mae'],'large_strata_deteriorate_at_most_5pct':large}
        if stage=='confirmation':checks['at_least_two_quarters_improve']=len(improved)>=THRESHOLDS['confirmation_min_improved_quarters']
    passed=all(checks.values())
    return {'passed':passed,'selected_candidate':CANDIDATE if passed else CONTROL,'checks':checks,'improved_quarters':improved,'candidate_reselection_performed':False}


def checked_path(name):
    path = (ROOT / Path(name)).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(), 'Missing/unsafe dependency: ' + str(name))
    return path


def require_no_staging(root=None):
    """An interrupted write is an error, never an unsuccessful candidate."""
    root = OUT if root is None else Path(root)
    for folder in (root / 'models', root / 'evaluation'):
        if folder.exists():
            unfinished = [p.name for p in folder.iterdir()
                          if p.name.startswith('.') or p.name.endswith(('.staging', '.partial', '.tmp'))]
            require(not unfinished, 'Unfinished experiment artifacts require review: ' + str(unfinished))


def geo_paths(stage):
    path = c6.OUT / 'geography' / stage / 'manifest.json'
    manifest = locked(path)
    paths = [path, path.with_suffix('.sha256')]
    paths += [checked_path(item['path']) for item in manifest['features'].values()]
    paths += [checked_path(name) for field in ('source_sha256', 'index_files_sha256') for name in manifest[field]]
    return sorted(set(paths))


def validate_geo_contract(stage, manifest, mapping):
    require(stage in STAGE_YEAR and manifest['stage'] == stage
            and manifest['snapshot_year'] == STAGE_YEAR[stage], 'Geographic stage/year differs')
    require(manifest['geography_provenance_chain']['verified'] is True, 'Unverified geographic provenance')
    require(manifest['feature_rules'] == component.base.FEATURE_RULES, 'Geographic rules changed')
    require(set(manifest['features']) == {'train', 'valid', 'test', 'full'}, 'Geographic inventory differs')
    before = datetime.fromisoformat(manifest['replication_timestamp'].replace('Z', '+00:00'))
    origin = datetime.fromisoformat(manifest['forecast_origin_utc'].replace('Z', '+00:00'))
    require(before < origin, 'Geographic map is not earlier than forecast origin')
    for field in ('source_sha256', 'index_files_sha256'):
        for name, digest in manifest[field].items():
            require(mapping.get(name) == digest, 'Unbound geographic dependency: ' + name)
    for item in manifest['features'].values():
        require(mapping.get(item['path']) == item['sha256'], 'Unbound encoded geographic frame')


def verify_geo_manifest(stage, specification):
    path = c6.OUT / 'geography' / stage / 'manifest.json'
    mapping = specification['final_dependency_sha256'] if stage == 'final' else specification['stage_dependency_sha256'][stage]
    require(mapping.get(rel(path)) == sha(path), 'Geographic manifest differs from frozen input')
    manifest = locked(path)
    validate_geo_contract(stage, manifest, mapping)
    c6.verify_hashes(mapping)
    return manifest



def validate_original_plan_binding(known, plan_digest, receipt_digest, receipt_plan_digest):
    """The prior source/data authority must be the already released R6 plan."""
    plan_name = 'outputs/round6/candidate_plan.json'
    receipt_name = 'outputs/round6/candidate_plan.sha256'
    require(known.get(plan_name) == plan_digest, 'Original R6 plan differs from released v3')
    require(receipt_plan_digest == plan_digest, 'Original R6 plan receipt content differs')
    if receipt_name in known:
        require(known[receipt_name] == receipt_digest, 'Original R6 receipt differs from released v3')


def verify_original_plan_snapshot(known):
    path = c6.OUT / 'candidate_plan.json'
    receipt = path.with_suffix('.sha256')
    plan_digest, receipt_digest = sha(path), sha(receipt)
    validate_original_plan_binding(known, plan_digest, receipt_digest,
                                   receipt.read_text(encoding='ascii').strip())
    return plan_digest, receipt_digest


def schedule_is_fixed(parameters, baseline):
    return (baseline.get('iterations') == 2999 and baseline.get('learning_rate') == 0.065
            and baseline.get('boosting_type') == 'Plain' and baseline.get('depth') == 8
            and parameters == baseline)


def research_bindings():
    manifest=ROOT/RESEARCH_FILES[0];value=locked(manifest)
    require(sha(manifest)=='a6aa73d6f06ad41131e8d4ecf7fc4ad4f11f57d915b0daf7f5c03a90ae708036'
            and value['round']==24 and value['status']=='source_only_method_review_complete','Reviewed method source manifest changed')
    mapping={rel(manifest):sha(manifest),rel(manifest.with_suffix('.sha256')):sha(manifest.with_suffix('.sha256'))}
    for field in ('source_sha256','evidence_sha256'):
        for name,digest in value.get(field,{}).items():
            require(name not in mapping or mapping[name]==digest,'Conflicting method evidence')
            mapping[name]=digest
    for item in value.get('inputs',{}).values():
        require(item['path'] not in mapping or mapping[item['path']]==item['sha256'],'Conflicting research input')
        mapping[item['path']]=item['sha256']
    require(len(mapping)>2,'Method evidence has no declared sources')
    require(not any(n.endswith('work_log.md') or n=='CURRENT_STATUS.md' for n in mapping),'Mutable document in model freeze')
    for name,digest in mapping.items():require(sha(checked_path(name))==digest,'Method source changed: '+name)
    return mapping


def plan_create():
    require_no_staging()
    path = OUT / 'candidate_plan.json'
    require(not path.exists() and not path.with_suffix('.sha256').exists(), 'R24 plan exists or is partial; never overwrite')
    require(sha(Path(component.__file__)) == HELPER_SHA, 'Reviewed fixed seed-ensemble helper changed')
    snapshot_path = ROOT / 'runs/v3/snapshot_manifest.json'
    require(sha(snapshot_path) == SNAPSHOT_SHA, 'Released v3 snapshot changed')
    snapshot = locked(snapshot_path)
    known = {k: v['original_sha256'] for k, v in snapshot['copied_files'].items()}
    for item in snapshot['external_dependencies']:
        name, digest = item['original_path'], item['sha256']
        require(name not in known or known[name] == digest, 'Conflicting snapshot identity: ' + name)
        known[name] = digest
    original_plan_binding = verify_original_plan_snapshot(known)
    old = c6.plan()
    require(verify_original_plan_snapshot(known) == original_plan_binding,
            'Original R6 plan/receipt changed while validating sources')
    dependencies = dict(old['data_sha256'])
    research = research_bindings()
    for name, digest in research.items():
        require(name not in dependencies or dependencies[name] == digest, 'Conflicting research dependency')
        dependencies[name] = digest
    required = [
        ROOT / 'outputs/submission_v3.csv', ROOT / 'outputs/submission_v2.csv',
        snapshot_path, snapshot_path.with_suffix('.sha256'),
        c6.OUT / 'candidate_plan.json', c6.OUT / 'candidate_plan.sha256',
        ROOT / 'runs/v1/models/final_config.json',
        OUT / 'audit/preflight.json', OUT / 'audit/preflight.sha256',
    ] + [ROOT / name for name in REQUIRED_DOCUMENTS]
    for item in required:
        dependencies[rel(item)] = sha(item)
    require(dependencies['outputs/submission_v3.csv'] == V3_SHA
            and dependencies['outputs/submission_v2.csv'] == V2_SHA, 'Preserved submissions changed')
    preflight = locked(OUT / 'audit/preflight.json')
    require(preflight['all_checks_passed'] is True
            and preflight['real_model_training_or_scoring'] is False, 'Synthetic preflight did not pass')
    captured_sources, captured_runtime = sources(), runtime_versions()
    require(preflight['source_sha256'] == captured_sources and preflight['runtime'] == captured_runtime,
            'Sources/runtime changed since preflight')
    stages = {}
    for stage in STAGES:
        mapping = {}
        for item in sorted(set(stage_paths(stage) + geo_paths(stage))):
            name, digest = rel(item), sha(item)
            require(known.get(name) == digest, 'Stage dependency differs from released v3: ' + name)
            mapping[name] = digest
        validate_geo_contract(stage, locked(c6.OUT / 'geography' / stage / 'manifest.json'), mapping)
        stages[stage] = mapping
    final_dependencies = {}
    final_paths = geo_paths('final') + [c6.OUT / 'release/release.json', c6.OUT / 'release/release.sha256']
    for item in sorted(set(final_paths)):
        name, digest = rel(item), sha(item)
        require(known.get(name) == digest, 'Final geographic dependency differs from v3: ' + name)
        final_dependencies[name] = digest
    validate_geo_contract('final', locked(c6.OUT / 'geography/final/manifest.json'), final_dependencies)
    value = {
        'round': 24, 'created_at_utc': now(), 'candidate': CANDIDATE, 'control': CONTROL,
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'parameters': component.parameters(42),
        'component': component.COMPONENT, 'target_mode': component.TARGET_MODE,
        'target_formula': component.TARGET_FORMULA, 'inverse_formula': component.INVERSE_FORMULA,
        'feature_count': 95, 'categorical_count': 23, 'feature_rules': component.FEATURE_RULES,
        'source_sha256': captured_sources, 'dependency_sha256': dependencies,
        'research_dependency_sha256': research,
        'stage_dependency_sha256': stages, 'final_dependency_sha256': final_dependencies,
        'final_contract': old['final_contract'], 'splits': old['splits'],
        'split_contract': old['split_contract'], 'target_group_weights': old['target_group_weights'],
        'quarter_definition': old['quarter_definition'], 'old_plan_sha256': sha(c6.OUT / 'candidate_plan.json'),
        'v3_snapshot_sha256': SNAPSHOT_SHA, 'reviewed_helper_sha256': HELPER_SHA,
        'single_development_candidate': True, 'tuning_or_weight_search': False,
        'seeds': list(component.SEEDS), 'trained_seeds': list(component.NEW_SEEDS), 'reused_seeds': [42],
        'member_parameters': {str(seed): component.parameters(seed) for seed in component.SEEDS},
        'ensemble_formula': component.ENSEMBLE_FORMULA, 'seed_or_member_score_selection': False,
        'baseline_parameters': component.base.parameters(), 'changed_parameter_names': [],
        'removed_feature_names': REMOVED_FEATURES, 'added_feature_names': ADDED_FEATURES,
        'baseline_feature_count': 95, 'original95_preserved': True,
        'capacity_constraint': {'depth': 8, 'max_leaves_per_tree': 256, 'grow_policy': 'SymmetricTree', 'boosting_type': 'Plain'},
        'native_learning_rate_contract': {'expected': 0.065, 'rtol': 1e-7, 'atol': 0.0},
        'sample_weight': 'uniform', 'query_labels_used_for_training': False,
        'shared_loader_deserializes_query_labels': True, 'query_labels_discarded_by_fit_caller': True,
        'geographic_feature_state_uses_query_or_y': False,
        'model_hyperparameter_change': {'new_member_random_seeds':[137,2026]},
        'member_changed_parameter_names':['random_seed'], 'original_features_and_scoring_keys_preserved': True,
        'weight_argument_passed_to_pool': False,
        'public_score_used_for_selection': False, 'historically_untouched': False,
        'external_property_answers_used': False, 'fit_uses_eval_set': False,
        'early_stopping_used': False, 'runtime': captured_runtime,
        'registry_sha256_at_plan': sha(ROOT / 'runs/versions.json'),
        'final_fit_implemented_in_this_driver': False,
        'negative_raw_model_output_is_diagnostic_only': True,
        'calibration_or_extra_price_cap': False,
    }
    c6.verify_hashes(dependencies)
    for mapping in (*stages.values(), final_dependencies):
        c6.verify_hashes(mapping)
    require(value['source_sha256'] == preflight['source_sha256'] == sources()
            and value['runtime'] == runtime_versions(), 'Sources/runtime changed while freezing plan')
    require(verify_original_plan_snapshot(known) == original_plan_binding, 'Original R6 plan/receipt changed before writing frozen plan')
    locked_write(path, value)
    print(json.dumps({'status': 'plan_frozen', 'plan_sha256': sha(path), 'candidate': CANDIDATE}), flush=True)


def plan():
    value = locked(OUT / 'candidate_plan.json')
    require(value['round'] == 24 and value['candidate'] == CANDIDATE and value['control'] == CONTROL,
            'Wrong R24 plan')
    require(value['formulas'] == FORMULAS and value['thresholds'] == THRESHOLDS
            and value['parameters'] == component.parameters(42), 'Fixed configuration changed')
    require(value['baseline_parameters'] == component.base.parameters()
            and value['changed_parameter_names'] == []
            and value['removed_feature_names'] == REMOVED_FEATURES
            and value['added_feature_names'] == ADDED_FEATURES
            and value['baseline_feature_count'] == 95 and value['original95_preserved'] is True
            and schedule_is_fixed(value['parameters'], value['baseline_parameters'])
            and value['capacity_constraint'] == {'depth': 8, 'max_leaves_per_tree': 256, 'grow_policy': 'SymmetricTree', 'boosting_type': 'Plain'},
            'Fixed fine-grid schedule changed')
    require(value['native_learning_rate_contract'] == {'expected': 0.065, 'rtol': 1e-7, 'atol': 0.0},
            'Native learning rate contract changed')
    require(value['feature_rules'] == component.FEATURE_RULES and value['feature_count'] == 95
            and value['categorical_count'] == 23, 'Original feature contract changed')
    require(value['component'] == component.COMPONENT and value['target_mode'] == component.TARGET_MODE
            and value['target_formula'] == component.TARGET_FORMULA
            and value['inverse_formula'] == component.INVERSE_FORMULA, 'Target/inverse contract changed')
    require(value['source_sha256'] == sources() and sha(Path(component.__file__)) == HELPER_SHA,
            'Frozen source closure changed')
    require(value['splits'] == {k: list(v) for k, v in c6.SPLITS.items()}, 'Split definitions changed')
    require(value['runtime'] == runtime_versions(), 'Runtime versions changed')
    require(value['sample_weight'] == 'uniform' and value['query_labels_used_for_training'] is False
            and value['geographic_feature_state_uses_query_or_y'] is False
            and value['model_hyperparameter_change'] == {'new_member_random_seeds':[137,2026]}
            and value['member_changed_parameter_names']==['random_seed']
            and value['weight_argument_passed_to_pool'] is False
            and value['original_features_and_scoring_keys_preserved'] is True, 'Data-use declarations differ')
    require(all(value['dependency_sha256'].get(k) == v for k, v in value['research_dependency_sha256'].items()),
            'Research dependencies are outside frozen input map')
    require(value['seeds']==list(component.SEEDS) and value['trained_seeds']==list(component.NEW_SEEDS)
            and value['reused_seeds']==[42] and value['ensemble_formula']==component.ENSEMBLE_FORMULA
            and value['member_parameters']=={str(seed):component.parameters(seed) for seed in component.SEEDS}
            and value['seed_or_member_score_selection'] is False, 'Fixed seed ensemble changed')
    require(value['research_dependency_sha256']==research_bindings(), 'Research mapping changed')
    c6.verify_hashes(value['dependency_sha256'])
    require(sha(ROOT / 'runs/versions.json') == value['registry_sha256_at_plan'], 'Preserved registry changed')
    return value


def validate_freeze(frozen, development, specification):
    require(frozen['round'] == 24 and frozen['candidate'] == CANDIDATE, 'Wrong development freeze')
    require(frozen['plan_sha256'] == sha(OUT / 'candidate_plan.json')
            and frozen['result_sha256'] == sha(OUT / 'evaluation/development/results.json'),
            'Development freeze identity differs')
    require(frozen['source_sha256'] == specification['source_sha256']
            and frozen['parameters'] == specification['parameters']
            and frozen['formulas'] == FORMULAS and frozen['thresholds'] == THRESHOLDS
            and frozen['target_group_weights'] == specification['target_group_weights'],
            'Frozen development protocol differs')
    require(frozen['passed'] == development['decision']['passed']
            and frozen['later_stages_allowed'] == development['decision']['passed']
            and frozen['selected_candidate'] == development['decision']['selected_candidate'],
            'Frozen selection differs from the saved development decision')
    c6.verify_hashes(frozen['development_artifact_sha256'])


def authorize(stage):
    require(stage != 'final', 'Final fit/release is not implemented by this backtest driver')
    require(stage in STAGES, 'Only the fixed backtest stages are supported')
    specification = plan()
    verify_geo_manifest(stage, specification)
    if stage != 'development':
        frozen = locked(OUT / 'frozen_candidate.json')
        check_later_gate(stage, frozen, sha(OUT / 'candidate_plan.json'))
        development = read_result('development', specification)
        validate_freeze(frozen, development, specification)
    return specification


def raw_diagnostics(z, predictions):
    z = np.asarray(z)
    predictions = np.asarray(predictions)
    require(z.dtype == np.dtype('float64') and z.ndim == 1 and len(z) > 0
            and np.isfinite(z).all(), 'Invalid raw million-TWD predictions')
    require(predictions.shape == z.shape and predictions.dtype == np.dtype('float64')
            and np.isfinite(predictions).all() and (predictions >= 0).all(), 'Invalid component TWD predictions')
    require(np.array_equal(component.inverse_transform(z), predictions), 'Raw inverse differs from component TWD')
    return {
        'rows': int(len(z)), 'negative_count': int(np.count_nonzero(z < 0)),
        'zero_count': int(np.count_nonzero(z == 0)), 'positive_count': int(np.count_nonzero(z > 0)),
        'min': float(z.min()), 'max': float(z.max()),
        'zero_price_count': int(np.count_nonzero(predictions == 0)),
        'raw_values_sha256': value_sha(z, '<f8'),
        'TWD_values_sha256': value_sha(predictions, '<f8'),
        'inverse_matches_component_all_rows': True, 'used_for_tuning': False,
    }


def raw_predict(model, state, X, geo, cats):
    component.check_model(model, state)
    component.check_state(state, X, cats)
    augmented = component.augment_features(X, geo)
    categories = component.categories(X, cats)
    require(list(model.feature_names_) == list(augmented.columns) == state['all_feature_names'],
            'Raw-output model schema differs')
    values = np.asarray(model.predict(Pool(augmented, cat_features=categories, thread_count=4),
                                      thread_count=4), dtype='float64')
    require(values.shape == (len(X),) and np.isfinite(values).all(), 'Nonfinite or misaligned raw predictions')
    return values


def feature_hashes(X, Xq, geo, geoq):
    return {name: frame_sha(frame) for name, frame in [
        ('fit_X_sha256', X), ('query_X_sha256', Xq),
        ('fit_geo_sha256', geo), ('query_geo_sha256', geoq)]}


def compare_reloaded_native_capacity(fitted, reloaded):
    """Keep seven non-LR native fields exact; allow only the fixed LR storage tolerance."""
    expected_keys = set(NATIVE_CAPACITY_FIELDS)
    require(isinstance(fitted, dict) and isinstance(reloaded, dict)
            and set(fitted) == set(reloaded) == expected_keys,
            'Fitted/reloaded native capacity field inventory differs')
    fitted_rate = float(fitted['actual_learning_rate'])
    reloaded_rate = float(reloaded['actual_learning_rate'])
    for rate in (fitted_rate, reloaded_rate):
        require(np.isfinite(rate) and np.isclose(rate, 0.065, rtol=1e-7, atol=0),
                'Fitted/reloaded native learning rate violates the fixed request')
    require(np.isclose(fitted_rate, reloaded_rate, rtol=1e-7, atol=0),
            'Reloaded native learning rate exceeds the fixed storage tolerance')
    require(all(fitted[key] == reloaded[key] for key in expected_keys
                if key != 'actual_learning_rate'),
            'Reloaded native capacity differs from fitted model')
    # The published model is the saved CBM; retain the fitted representation separately.
    return {**reloaded, 'actual_learning_rate': reloaded_rate,
            'fitted_actual_learning_rate': fitted_rate}


def validate_model_contract(meta,specification,expected):
    require(meta['round']==24 and meta['family']=='catboost_seed_ensemble' and meta['component']==component.COMPONENT,
            'Wrong seed ensemble')
    require(meta['seeds']==[42,137,2026] and meta['trained_seeds']==[137,2026] and meta['reused_seeds']==[42]
            and meta['ensemble_formula']==component.ENSEMBLE_FORMULA,'Wrong members or averaging order')
    require(meta['parameters']==specification['parameters']==component.parameters(42)
            and meta['member_parameters']==specification['member_parameters'],'Wrong seed recipes')
    require(meta['feature_count']==95 and meta['categorical_feature_count']==23 and meta['iterations_per_member']==2999
            and meta['sample_weight']=='uniform' and meta['target_mode']=='raw' and meta['prediction_units']=='TWD',
            'Wrong original95/raw/uniform recipe')
    require(meta['all_members_replayed_all_query_rows'] is True and meta['mean_exact_equality'] is True
            and meta['seed42_retrained'] is False and meta['individual_member_scores_computed'] is False,
            'Incomplete replay or seed selection')
    require(set(meta['member_metadata_sha256'])=={'137','2026'} and set(meta['artifact_sha256'])==MODEL_FILES,'Incomplete ensemble files')
    for key,value in expected.items():require(meta.get(key)==value,'Ensemble metadata differs: '+key)


def check_model_files(stage,specification):
    folder=OUT/'models'/stage;meta=locked(folder/'metadata.json')
    validate_model_contract(meta,specification,stage_alignment(stage,specification))
    files={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(files==MODEL_FILES|{'metadata.json','metadata.sha256'},'Incomplete/extra ensemble inventory')
    for name,digest in meta['artifact_sha256'].items():
        require(name in MODEL_FILES and sha(folder/name)==digest,'Saved ensemble artifact changed')
    for seed in component.NEW_SEEDS:
        branch=folder/f'seed_{seed}';member=locked(branch/'metadata.json');member_contract(member,seed,specification)
        require(meta['member_metadata_sha256'][str(seed)]==sha(branch/'metadata.json'),'Member metadata hash differs')
        for key,value in stage_alignment(stage,specification).items():require(member[key]==value,'Member stage alignment differs: '+key)
        require(set(member['artifact_sha256'])==BRANCH_FILES-{'metadata.json','metadata.sha256'},'Member file inventory differs')
        for name,digest in member['artifact_sha256'].items():require(sha(branch/name)==digest,'Member artifact changed')
    return meta,{rel(p):sha(p) for p in folder.rglob('*') if p.is_file()}




def stage_alignment(stage,specification):
    split=specification['split_contract'][stage]
    return {'stage':stage,'plan_sha256':sha(OUT/'candidate_plan.json'),'source_sha256':specification['source_sha256'],
        'outer_fit_rows':split['fit_rows'],'query_rows':split['query_rows'],
        'fit_indices_content_sha256':split['fit_indices_content_sha256'],
        'query_indices_content_sha256':split['query_indices_content_sha256'],
        'geo_manifest_sha256':sha(c6.OUT/'geography'/stage/'manifest.json'),
        'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json')}

def member_contract(meta,seed,specification):
    require(meta['source_sha256']==specification['source_sha256'] and meta['feature_rules']==component.FEATURE_RULES,
            'Member source/rules changed')
    require(meta['seed']==seed and seed in component.NEW_SEEDS and meta['component']==component.COMPONENT,
            'Wrong new member')
    require(meta['parameters']==component.parameters(seed)==specification['member_parameters'][str(seed)]
            and meta['baseline_parameters']==specification['baseline_parameters']
            and meta['changed_parameter_names']==['random_seed'],'Member differs beyond fixed seed')
    require(meta['feature_count']==95 and meta['categorical_feature_count']==meta['native_categorical_feature_count']==23
            and meta['actual_iterations']==meta['actual_trees']==2999 and meta['actual_random_seed']==seed,
            'Wrong member capacity/seed')
    require(meta['target_formula']==component.TARGET_FORMULA and meta['inverse_formula']==component.INVERSE_FORMULA
            and meta['sample_weight']=='uniform' and meta['weight_argument_passed_to_pool'] is False,'Wrong member target/weight')
    for key in ('fit_uses_eval_set','early_stopping_used','query_labels_used','target_used_for_geographic_state',
                'query_used_for_geographic_state','reference_or_trend_applied_here','calibration_applied','extra_price_cap_applied'):
        require(meta[key] is False,'Forbidden member behavior: '+key)
    require(meta['reload_all_query_rows_verified'] is True and meta['reload_raw_z_exact_equality'] is True,
            'New member full-query replay incomplete')

def replay_seed42(stage,X,y,Xq,fi,qi,geo,geoq,cats,specification):
    folder=c6.OUT/'models'/stage/'geography_catboost'
    names=('model.cbm','feature_state.json','metadata.json','predictions.npy','fit_indices.npy','query_indices.npy')
    deps={rel(folder/n):sha(folder/n) for n in names}
    require(all(specification['stage_dependency_sha256'][stage].get(n)==d for n,d in deps.items()),'Seed42 artifact not snapshot-bound')
    e6.verify_indices(folder,fi,qi,'query_indices.npy')
    meta=read(folder/'metadata.json');state=read(folder/'feature_state.json')
    require(meta['component']=='geography_catboost' and meta['parameters']==component.parameters(42)
            and meta['target_mode']=='raw' and meta['sample_weight']=='uniform'
            and meta['fit_target_values_sha256']==value_sha(y,'<f8'),'Original seed42 model recipe/target differs')
    require(state==component.base.fit_feature_state(X,geo),'Original flat seed42 state differs from fit-only reconstruction')
    component.base.check_state(state,Xq)
    model=load_model(folder/'model.cbm');augmented=component.augment_features(Xq,geoq);categories=component.categories(Xq,cats)
    native=component.check_native_model(model,list(augmented.columns),categories,42)
    z=np.asarray(model.predict(Pool(augmented,cat_features=categories,thread_count=4),thread_count=4),dtype='float64')
    predictions=component.inverse_transform(z);original=e6.vector(folder/'predictions.npy',len(qi))
    require(np.array_equal(predictions,original),'Saved seed42 full-query replay differs from original prediction')
    evidence={'round':24,'stage':stage,'seed':42,'retrained':False,'source_artifact_sha256':deps,
        'original_state_is_flat_R6':True,'flat_state_content_sha256':state['state_content_sha256'],
        'fit_state_reconstructed_from_fit_X_geo_only':True,'all_query_rows_replayed':True,'query_rows':len(qi),
        'original_saved_predictions_bitwise_equal':True,'raw_values_sha256':value_sha(z,'<f8'),
        'TWD_values_sha256':value_sha(predictions,'<f8'),'native_model_capacity':native,
        'native_parameters':model.get_all_params(),'original_metadata_reload_64_rows_verified':meta.get('reload_64_rows_verified'),
        'this_round_full_replay_is_separate_from_original_64_row_check':True,
        'fit_indices_content_sha256':value_sha(fi,'<i8'),'query_indices_content_sha256':value_sha(qi,'<i8')}
    return predictions,z,evidence

def fit(stage):
    require_no_staging();specification=authorize(stage);destination=OUT/'models'/stage
    require(not destination.exists(),'Ensemble already exists; never retrain')
    data,X,y,Xq,yq,fi,qi=partition(stage);del yq
    geo,geoq,_=load_geographic_partition(stage,fi,qi);cats=data['categorical_columns']
    before=feature_hashes(X,Xq,geo,geoq);started=time.monotonic()
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    predictions={};predictions[42],z42,evidence=replay_seed42(stage,X,y,Xq,fi,qi,geo,geoq,cats,specification)
    np.save(staging/'seed42_raw_z.npy',z42,allow_pickle=False);np.save(staging/'seed42_predictions.npy',predictions[42],allow_pickle=False)
    locked_write(staging/'seed42_replay.json',evidence)
    member_hashes={}
    for seed in component.NEW_SEEDS:
        branch=staging/f'seed_{seed}';branch.mkdir()
        print(json.dumps({'event':'member_fit_started','stage':stage,'seed':seed,'fit_rows':len(X),'query_rows':len(Xq)}),flush=True)
        model,state,meta=component.fit_model(X,y,geo,cats,seed)
        z=raw_predict(model,state,Xq,geoq,cats);prediction=component.inverse_transform(z)
        model.save_model(str(branch/'model.cbm'));write(branch/'feature_state.json',state)
        loaded=load_model(branch/'model.cbm');restored=read(branch/'feature_state.json')
        z_replay=raw_predict(loaded,restored,Xq,geoq,cats);replay=component.predict_model(loaded,restored,Xq,geoq,cats)
        require(np.array_equal(z,z_replay) and np.array_equal(prediction,replay),'New member full-query saved replay differs')
        meta.update(compare_reloaded_native_capacity({k:meta[k] for k in NATIVE_CAPACITY_FIELDS},component.check_model(loaded,restored)))
        for name,values in [('raw_z',z),('predictions',prediction),('fit_indices',fi),('query_indices',qi)]:np.save(branch/(name+'.npy'),values,allow_pickle=False)
        meta.update({'round':24,**stage_alignment(stage,specification),'component_source_sha256':meta['source_sha256'],
            'source_sha256':specification['source_sha256'],'reload_all_query_rows_verified':True,
            'reload_raw_z_exact_equality':True,'reload_query_rows':len(qi),'raw_z_diagnostics':raw_diagnostics(z,prediction),
            'artifact_sha256':{p.name:sha(p) for p in branch.iterdir() if p.is_file()},**before})
        member_contract(meta,seed,specification);locked_write(branch/'metadata.json',meta)
        member_hashes[str(seed)]=sha(branch/'metadata.json');predictions[seed]=prediction
        print(json.dumps({'event':'member_fit_completed','stage':stage,'seed':seed,'all_query_replay_exact':True}),flush=True)
        del model,loaded
    mean=component.average_predictions(predictions)
    for name,values in [('predictions',mean),('fit_indices',fi),('query_indices',qi)]:np.save(staging/(name+'.npy'),values,allow_pickle=False)
    require(np.array_equal(mean,np.load(staging/'predictions.npy',allow_pickle=False)), 'Written mean prediction differs')
    require(before==feature_hashes(X,Xq,geo,geoq),'Ensemble changed original X/geo')
    meta={'round':24,'family':'catboost_seed_ensemble','component':component.COMPONENT,**stage_alignment(stage,specification),
        'created_at_utc':now(),'seeds':list(component.SEEDS),'trained_seeds':list(component.NEW_SEEDS),'reused_seeds':[42],
        'ensemble_formula':component.ENSEMBLE_FORMULA,'parameters':specification['parameters'],'member_parameters':specification['member_parameters'],
        'feature_count':95,'categorical_feature_count':23,'iterations_per_member':2999,'target_mode':'raw','sample_weight':'uniform',
        'prediction_units':'TWD','all_members_replayed_all_query_rows':True,'mean_exact_equality':True,
        'seed42_retrained':False,'individual_member_scores_computed':False,'member_metadata_sha256':member_hashes,
        'fit_target_sha256':value_sha(y,'<f8'),**before,'feature_rules':component.FEATURE_RULES,
        'artifact_sha256':{p.relative_to(staging).as_posix():sha(p) for p in staging.rglob('*') if p.is_file()},
        'shared_loader_deserialized_query_labels':True,'query_labels_discarded_by_fit_caller':True,'query_labels_used':False,
        'runner_elapsed_seconds':time.monotonic()-started}
    validate_model_contract(meta,specification,before);locked_write(staging/'metadata.json',meta)
    authorize(stage);publish(staging,destination)
    print(json.dumps({'event':'fit_completed','stage':stage,'trained_seeds':list(component.NEW_SEEDS),
                      'reused_seeds':[42],'elapsed_seconds':time.monotonic()-started}),flush=True)


def model_check(stage,X,y,Xq,fi,qi,geo,geoq,cats,specification):
    meta,inputs=check_model_files(stage,specification);folder=OUT/'models'/stage
    before=feature_hashes(X,Xq,geo,geoq);validate_model_contract(meta,specification,before)
    require(meta['fit_target_sha256']==value_sha(y,'<f8'),'Ensemble fit targets differ')
    e6.verify_indices(folder,fi,qi,'query_indices.npy')
    predictions={};predictions[42],z42,evidence=replay_seed42(stage,X,y,Xq,fi,qi,geo,geoq,cats,specification)
    require(evidence==locked(folder/'seed42_replay.json'),'Seed42 replay evidence differs')
    require(np.array_equal(z42,np.load(folder/'seed42_raw_z.npy',allow_pickle=False))
            and np.array_equal(predictions[42],e6.vector(folder/'seed42_predictions.npy',len(qi))),'Seed42 saved vectors differ')
    for seed in component.NEW_SEEDS:
        branch=folder/f'seed_{seed}';member=locked(branch/'metadata.json');state=read(branch/'feature_state.json')
        require(state==component.fit_feature_state(X,geo,cats,seed),'New member fit-only state differs')
        require(member['fit_original_target_values_sha256']==value_sha(y,'<f8')
                and member['fit_transformed_target_values_sha256']==value_sha(component.transform_target(y),'<f8'),'Member target identity differs')
        e6.verify_indices(branch,fi,qi,'query_indices.npy');model=load_model(branch/'model.cbm')
        require(all(member[k]==v for k,v in component.check_model(model,state).items()),'Saved member native diagnostics differ')
        z=raw_predict(model,state,Xq,geoq,cats);prediction=component.predict_model(model,state,Xq,geoq,cats)
        require(np.array_equal(z,np.load(branch/'raw_z.npy',allow_pickle=False))
                and np.array_equal(prediction,e6.vector(branch/'predictions.npy',len(qi))),'New member replay differs')
        require(raw_diagnostics(z,prediction)==member['raw_z_diagnostics'],'Member diagnostics differ')
        predictions[seed]=prediction;del model
    mean=component.average_predictions(predictions);saved=e6.vector(folder/'predictions.npy',len(qi))
    require(np.array_equal(mean,saved),'Saved equal-weight mean differs')
    require(before==feature_hashes(X,Xq,geo,geoq),'Ensemble replay changed X/geo')
    return saved,inputs


def validate_result_contract(stage, value, specification):
    require(value['round'] == 24 and value['stage'] == stage
            and value['plan_sha256'] == sha(OUT / 'candidate_plan.json'), 'Wrong saved result')
    require(value['source_sha256'] == specification['source_sha256'] and value['formulas'] == FORMULAS
            and value['thresholds'] == THRESHOLDS, 'Saved result protocol differs')
    require(value['frozen_candidate_sha256'] == (None if stage == 'development' else sha(OUT / 'frozen_candidate.json')),
            'Saved result freeze differs')
    require(value['evaluated_candidates'] == [CONTROL, CANDIDATE]
            and value['candidate_component_separately_scored'] is False, 'Candidate inventory differs')
    require(value['query_labels_used_for_training'] is False
            and value['historically_untouched'] is False, 'Saved data-use claims differ')
    require(value['fit_rows'] == specification['split_contract'][stage]['fit_rows']
            and value['query_rows'] == specification['split_contract'][stage]['query_rows'],
            'Saved sample counts differ')
    require(value['target_group_weights'] == specification['target_group_weights'], 'Scoring weights differ')
    require(value['decision'] == decide(stage, value['metrics']), 'Saved gate decision differs')


def read_result(stage, specification=None):
    require(stage in STAGES, 'Unknown saved result stage')
    specification = plan() if specification is None else specification
    folder = OUT / 'evaluation' / stage
    value = locked(folder / 'results.json')
    validate_result_contract(stage, value, specification)
    _, model_inputs = check_model_files(stage, specification)
    for name, digest in {**model_inputs, **specification['stage_dependency_sha256'][stage]}.items():
        require(value['input_artifact_sha256'].get(name) == digest, 'Result is missing a model/stage input')
    require(set(value['output_artifact_sha256']) == {rel(folder / (name + '.npy')) for name in (CONTROL, CANDIDATE)},
            'Saved prediction inventory differs')
    require({p.name for p in folder.iterdir()} == {CONTROL + '.npy', CANDIDATE + '.npy', 'results.json', 'results.sha256'},
            'Result folder has incomplete or unexpected files')
    for field in ('input_artifact_sha256', 'output_artifact_sha256'):
        c6.verify_hashes(value[field])
    return value


def freeze_development(result):
    require_no_staging()
    specification = plan()
    require(read_result('development', specification) == result, 'Only the saved validated result may be frozen')
    artifacts = {rel(p): sha(p) for folder in (OUT / 'models/development', OUT / 'evaluation/development')
                 for p in folder.iterdir() if p.is_file()}
    value = {
        'created_at_utc': now(), 'round': 24, 'candidate': CANDIDATE,
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'result_sha256': sha(OUT / 'evaluation/development/results.json'),
        'passed': result['decision']['passed'], 'selected_candidate': result['decision']['selected_candidate'],
        'development_artifact_sha256': artifacts, 'source_sha256': specification['source_sha256'],
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'parameters': component.parameters(42),
        'target_group_weights': specification['target_group_weights'],
        'later_stages_allowed': result['decision']['passed'], 'historically_untouched': False,
    }
    path = OUT / 'frozen_candidate.json'
    if path.exists():
        saved = locked(path)
        require(all(saved[key] == item for key, item in value.items() if key != 'created_at_utc'),
                'Existing development freeze differs')
    else:
        locked_write(path, value)


def evaluate(stage):
    require_no_staging()
    specification = authorize(stage)
    destination = OUT / 'evaluation' / stage
    if destination.exists():
        result = read_result(stage, specification)
        if stage == 'development':
            freeze_development(result)
        print(json.dumps({'status': 'existing_result_verified', 'stage': stage, 'decision': result['decision']}), flush=True)
        return result
    data, X, y, Xq, yq, fi, qi = partition(stage)
    weights = target_weights(data)
    require(weights == specification['target_group_weights'], 'Fixed test-X grouping weights differ')
    geo, geoq, _ = load_geographic_partition(stage, fi, qi)
    new, inputs = model_check(stage, X, y, Xq, fi, qi, geo, geoq, data['categorical_columns'], specification)
    inputs.update(specification['stage_dependency_sha256'][stage])
    reference, _ = c6.load_reference(stage, X)
    base = c6.baseline_folder(stage)
    old = e6.vector(base / 'catboost_raw.npy', len(qi))
    unit = e6.vector(base / 'lightgbm_l1_unit63.npy', len(qi))
    geographic = e6.vector(c6.OUT / 'models' / stage / 'geography_catboost/predictions.npy', len(qi))
    original = e6.vector(c6.OUT / 'evaluation' / stage / 'geography_blend.npy', len(qi))
    replay = blend(old, geographic, unit, reference, Xq)
    require(np.allclose(original, replay, rtol=1e-12, atol=1e-5), 'v3 control does not replay')
    predictions = {CONTROL: original, CANDIDATE: blend(old, new, unit, reference, Xq)}
    metrics = {name: score(yq, pred, X, Xq, c6.SPLITS[stage][0], weights)
               for name, pred in predictions.items()}
    previous = locked(c6.OUT / 'evaluation' / stage / 'results.json')
    require(metrics[CONTROL] == previous['metrics']['geography_blend'], 'v3 score differs from archived same-window result')
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + stage + '-' + uuid.uuid4().hex)
    staging.mkdir()
    for name, values in predictions.items():
        np.save(staging / (name + '.npy'), values, allow_pickle=False)
    outputs = {rel(destination / p.name): sha(p) for p in staging.iterdir() if p.is_file()}
    result = {
        'round': 24, 'stage': stage, 'created_at_utc': now(),
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'frozen_candidate_sha256': None if stage == 'development' else sha(OUT / 'frozen_candidate.json'),
        'source_sha256': specification['source_sha256'], 'input_artifact_sha256': inputs,
        'output_artifact_sha256': outputs, 'formulas': FORMULAS, 'thresholds': THRESHOLDS,
        'target_group_weights': weights, 'metrics': metrics, 'decision': decide(stage, metrics),
        'v3_control_replayed_all_rows': True, 'new_model_replayed_all_rows': True,
        'new_model_raw_outputs_replayed_all_rows': True,
        'fit_rows': len(fi), 'query_rows': len(qi), 'evaluated_candidates': [CONTROL, CANDIDATE],
        'candidate_component_separately_scored': False, 'query_labels_used_for_training': False,
        'shared_loader_deserializes_query_labels': True, 'public_score_used': False,
        'historically_untouched': False, 'new_public_score_known': False, 'private_score_known': False,
    }
    locked_write(staging / 'results.json', result)
    authorize(stage)
    c6.verify_hashes(inputs)
    publish(staging, destination)
    if stage == 'development':
        freeze_development(result)
    print(json.dumps({'stage': stage,
                      'metrics': {name: {key: item[key] for key in ('overall_mae', 'weighted_mae')}
                                  for name, item in metrics.items()},
                      'decision': result['decision']}), flush=True)
    return result


def final_outcome(results):
    require('development' in results, 'Development result is required')
    if not results['development']['decision']['passed']:
        require(set(results) == {'development'}, 'Rejected development has later results')
        return False, 'development_price_gates_failed'
    require(set(results) == set(STAGES), 'Missing later results are incomplete, not rejection')
    if all(value['decision']['passed'] for value in results.values()):
        return True, 'all_fixed_backtests_passed'
    return False, 'later_backtest_gates_failed'


def finalize():
    require_no_staging()
    specification = plan()
    frozen = locked(OUT / 'frozen_candidate.json')
    development = read_result('development', specification)
    validate_freeze(frozen, development, specification)
    results = {'development': development}
    statuses = {'development': 'price_evaluated'}
    for stage in ('confirmation', 'auxiliary'):
        if frozen['passed']:
            authorize(stage)
            results[stage] = read_result(stage, specification)
            statuses[stage] = 'price_evaluated'
        else:
            require(not (OUT / 'models' / stage).exists() and not (OUT / 'evaluation' / stage).exists(),
                    'Rejected development has unauthorized later artifacts')
            statuses[stage] = 'not_executed_after_development_rejection'
    accepted, reason = final_outcome(results)
    plan()
    result = {
        'created_at_utc': now(), 'round': 24, 'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'frozen_candidate_sha256': sha(OUT / 'frozen_candidate.json'),
        'accepted': accepted, 'candidate': CANDIDATE, 'selected_candidate': frozen['selected_candidate'],
        'deployment_decision': 'backtest_accepted_pending_final_release' if accepted else 'keep_v3',
        'stopping_reason': reason, 'stage_status': statuses,
        'stage_result_sha256': {stage: sha(OUT / 'evaluation' / stage / 'results.json') for stage in results},
        'unevaluated_stages': [stage for stage in STAGES if stage not in results],
        'no_reselection_or_parameter_change': True, 'new_submission_created': False,
        'currently_recommended_submission': 'outputs/submission_v3.csv',
        'private_score_known': False, 'historically_untouched': False,
        'source_sha256': specification['source_sha256'],
    }
    path = OUT / 'accepted_candidate.json'
    if path.exists():
        saved = locked(path)
        require(all(saved[key] == value for key, value in result.items() if key != 'created_at_utc'),
                'Existing final decision differs')
        result = saved
    else:
        locked_write(path, result)
    print(json.dumps(result), flush=True)
    return result



def self_check(write_output=True):
    """Only new ensemble integration checks; reuse recorded unchanged helper run."""
    checks=[]
    def good(name,condition):require(condition,name);checks.append(name)
    def bad(name,fn):
        try:fn()
        except (ValueError,RuntimeError,KeyError,TypeError):checks.append(name);return
        raise AssertionError('Expected refusal: '+name)
    captured_sources=sources();captured_runtime=runtime_versions()
    require(sha(Path(component.__file__))==HELPER_SHA,'Previously tested helper source changed')
    spy=[]
    def trend(ref,X,base,variant):spy.append((ref,variant,np.array(base,copy=True)));return np.asarray(base)*1.1
    arrays={42:np.array([1e6,2e6]),137:np.array([2e6,3e6]),2026:np.array([3e6,4e6])}
    mean=component.average_predictions(arrays);Xq=[None,None]
    pred=blend(np.array([4e6,5e6]),mean,np.array([6e6,7e6]),'ref',Xq,trend)
    good('single_reference_after_TWD_seed_mean',len(spy)==1 and spy[0][:2]==('ref','trend')
         and np.array_equal(spy[0][2],.75*(.5*np.array([4e6,5e6])+.5*mean)+.25*np.array([6e6,7e6])))
    good('exact_old_outer_weights',np.array_equal(pred,spy[0][2]*1.1))
    native={'actual_model_depth':8,'actual_boosting_type':'Plain','actual_grow_policy':'SymmetricTree',
            'actual_trees':2999,'maximum_tree_leaf_count':256,'tree_leaf_counts_sha256':'a'*64,
            'actual_learning_rate':float(np.float32(.065)),'actual_random_seed':137}
    restored=dict(native,actual_learning_rate=.06499999762)
    result=compare_reloaded_native_capacity(native,restored)
    good('saved_LR_canonical_fixed_tolerance',result['actual_learning_rate']==restored['actual_learning_rate']
         and result['fitted_actual_learning_rate']==native['actual_learning_rate'])
    bad('reload_wrong_seed_rejected',lambda:compare_reloaded_native_capacity(native,dict(restored,actual_random_seed=2026)))
    bad('reload_wrong_LR_rejected',lambda:compare_reloaded_native_capacity(native,dict(restored,actual_learning_rate=.0325)))
    reject={'passed':False,'selected_candidate':CONTROL,'later_stages_allowed':False,'plan_sha256':'a'*64}
    accept={'passed':True,'selected_candidate':CANDIDATE,'later_stages_allowed':True,'plan_sha256':'a'*64}
    for stage in ('confirmation','auxiliary'):
        bad('later_blocked_'+stage,lambda stage=stage:check_later_gate(stage,reject,'a'*64))
        check_later_gate(stage,accept,'a'*64);checks.append('later_requires_own_pass_'+stage)
    bad('final_not_implemented',lambda:check_later_gate('final',accept,'a'*64))
    good('fixed_new_member_inventory',component.NEW_SEEDS==(137,2026) and component.SEEDS==(42,137,2026)
         and set(n.split('/')[0] for n in MODEL_FILES if '/' in n)=={'seed_137','seed_2026'})
    good('no_seed42_new_CBM',not any(n.startswith('seed_42/') for n in MODEL_FILES))
    old=component.base.parameters
    fake={'iterations':2999,'learning_rate':.065,'thread_count':4,'depth':8,'random_seed':42,
          'loss_function':'MAE','eval_metric':'MAE','boosting_type':'Plain','task_type':'GPU'}
    try:
        component.base.parameters=lambda:dict(fake)
        spec={'parameters':component.parameters(42),'member_parameters':{str(k):component.parameters(k) for k in component.SEEDS}}
        meta={'round':24,'family':'catboost_seed_ensemble','component':component.COMPONENT,
              'seeds':[42,137,2026],'trained_seeds':[137,2026],'reused_seeds':[42],
              'ensemble_formula':component.ENSEMBLE_FORMULA,**spec,'feature_count':95,'categorical_feature_count':23,
              'iterations_per_member':2999,'sample_weight':'uniform','target_mode':'raw','prediction_units':'TWD',
              'all_members_replayed_all_query_rows':True,'mean_exact_equality':True,'seed42_retrained':False,
              'individual_member_scores_computed':False,'member_metadata_sha256':{'137':'a'*64,'2026':'b'*64},
              'artifact_sha256':{n:'c'*64 for n in MODEL_FILES}}
        validate_model_contract(meta,spec,{});checks.append('complete_ensemble_metadata_contract')
        for key,value in [('trained_seeds',[42,137,2026]),('individual_member_scores_computed',True),
                          ('seed42_retrained',True),('all_members_replayed_all_query_rows',False),('feature_count',97)]:
            bad('metadata_reject_'+key,lambda key=key,value=value:validate_model_contract(dict(meta,**{key:value}),spec,{}))
    finally:component.base.parameters=old
    current=Path(__file__).read_text(encoding='utf-8-sig');tree=ast.parse(current)
    funcs={n.name:ast.get_source_segment(current,n) for n in tree.body if isinstance(n,ast.FunctionDef)}
    original=Path(ROOT/'src/round21_experiment.py').read_text(encoding='utf-8-sig')
    oldfuncs={n.name:ast.get_source_segment(original,n) for n in ast.parse(original).body if isinstance(n,ast.FunctionDef)}
    for name in ('blend','decide','check_later_gate','final_outcome'):
        require(ast.dump(ast.parse(funcs[name]),include_attributes=False)==ast.dump(ast.parse(oldfuncs[name]),include_attributes=False),
                'Unchanged gate/blend function changed: '+name)
    checks.append('four_existing_gate_blend_functions_AST_unchanged')
    good('seed42_CBM_state_are_frozen_stage_dependencies',"'model.cbm','feature_state.json','metadata.json'" in funcs['stage_paths'])
    good('seed42_flat_fit_state_and_all_query_equality',"component.base.fit_feature_state(X,geo)" in funcs['replay_seed42']
         and "np.array_equal(predictions,original)" in funcs['replay_seed42'])
    good('new_members_no_seed_search_no_single_scores',"for seed in component.NEW_SEEDS" in funcs['fit']
         and "component.fit_model(X,y,geo,cats,seed)" in funcs['fit'] and "'individual_member_scores_computed':False" in funcs['fit'])
    good('all_raw_and_price_saved_reload_before_publication',"np.array_equal(z,z_replay)" in funcs['fit']
         and "np.array_equal(prediction,replay)" in funcs['fit'])
    good('actual_mean_saved_and_replayed',"np.array_equal(mean,saved)" in funcs['model_check'])
    require(captured_sources==sources() and captured_runtime==runtime_versions(),'Sources/runtime changed during preflight')
    value={'round':24,'recorded_at_utc':now(),'all_checks_passed':True,'real_model_training_or_scoring':False,
           'source_sha256':captured_sources,'runtime':captured_runtime,'checks':checks,'check_count':len(checks),
           'component_synthetic_checks':{'check_count':45,'all_checks_passed':True,'real_training_or_config_or_data_read':False,
               'helper_sha256':HELPER_SHA,'execution':'completed once before driver finalization; recorded from actual exit0 tool output',
               'rerun_in_this_preflight':False},
           'component_source_unchanged_since_prior_checks':True,'no_formal_price_candidate_fitted':True}
    if write_output:locked_write(OUT/'audit/preflight.json',value)
    print(json.dumps(value,ensure_ascii=False),flush=True);return value


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['self-check', 'plan', 'fit', 'evaluate', 'finalize'])
    parser.add_argument('--stage', choices=(*STAGES, 'final'), default='development')
    parser.add_argument('--no-write', action='store_true', help='Synthetic only; do not publish preflight')
    args = parser.parse_args()
    if args.command == 'self-check':
        self_check(write_output=not args.no_write)
    elif args.command == 'plan':
        plan_create()
    elif args.command == 'fit':
        fit(args.stage)
    elif args.command == 'evaluate':
        evaluate(args.stage)
    else:
        finalize()

