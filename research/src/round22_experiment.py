"""R22: one fixed Huber objective for the original geographic CatBoost slot of v3.
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
import round22_huber_geo as component
from round6_prepare_geo import load_geographic_partition
from round6_geo import frame_sha
from round2_neighborhood import predict_reference
from round3_evaluate import score, target_weights, runtime_versions

ROOT = c6.ROOT
OUT = ROOT / 'outputs/round22'
STAGES = tuple(c6.SPLITS)
STAGE_YEAR = {'development': 2022, 'confirmation': 2023, 'auxiliary': 2021, 'final': 2024}
CANDIDATE = 'huber_catboost_geography_blend'
CONTROL = 'v3_control'
EXTRA_SOURCES = ()
REQUIRED_DOCUMENTS = (
    'reports/round22_plan.md',
    'reports/round22_method_review.md',
    'reports/round22_pipeline_review.md',
)
REMOVED_FEATURES = []
ADDED_FEATURES = []
CHANGED_PARAMETERS = ['loss_function']
NATIVE_OBJECTIVE_CONTRACT = {'loss_function': 'Huber:delta=1', 'eval_metric': 'MAE', 'leaf_estimation_method': 'Newton', 'leaf_estimation_iterations': 1, 'boost_from_average': False}
IMPLICIT_NATIVE_POLICY_CHANGES = {'boost_from_average': {'from': True, 'to': False}, 'leaf_estimation_method': {'from': 'Exact', 'to': 'Newton'}}
# Frozen source-only method review and synthetic GPU compatibility identities.
RESEARCH_EXPECTED_SHA = {'outputs/round22_research/huber_candidate.cbm': 'da11500536aaca3a6e932dd67e21e9bf604f35913b55812ace89f79c07cd0ba6', 'outputs/round22_research/huber_feasibility.json': '167fbc4122a75209b66527fd5e518de5b33626f2fc564b3903976ff403169ee1', 'outputs/round22_research/huber_feasibility.log': '83248d91f7da3fcb2dea1642b941fcf823b75ea447047ccf9c94b4770157b5a7', 'outputs/round22_research/huber_feasibility.sha256': '6cb3a168878033539792ce5e66cee5bf0ee2d95503d5bca2e6d7c150f22ffa85', 'outputs/round22_research/mae_control.cbm': '9e702b2cd86ea18465280aa67c85548b4bebfdc4a83f7905e83e66c2a3eb842e', 'outputs/round22_research/method_sources.json': '2194d3ef4ef04365e144625b6a022211bc057579486a44800e4d1de375897be4', 'outputs/round22_research/method_sources.sha256': '7a84fa6dd776cdf7606c2efb1eaa1c11920eb7898905e934cedc4c2688ba620e', 'reports/round21_plan.md': 'ca7929f4e30aa44880295260cd200c1fbe17fcc3a01960828560d8f0844833c2', 'reports/round22_method_review.md': 'c2a76058138a4e350e61a6257db91b3c759f223ad4ca00ecd8bc5de2d4d2f7da', 'runs/v1/models/final_config.json': '2ca4258f363d1937df50aed72f5b11dc32b7e70d9dd5fb57fb6f375673a2a425', 'src/prepare_data.py': '1b2a2c4c947aebdeff5d268300fac6f2f56331a850ffcc6eb6149e7f255ebde6', 'src/round12_log_geo.py': 'e16fbe33928bd1a635e5de8bd717f223263847b718aef373fb933224b13d6c42', 'src/round13_city_alias_geo.py': 'aed4d22c3bb911d163d42a3bef4eb47038dfd7ac8f34aec5c460da7ac57f6c42', 'src/round14_geo_raw_lgb.py': 'a22186c6a336d111868f51f83979ccdbac2f989d02b695e237464c5517e02d1d', 'src/round15_shallow_geo.py': '347573e96521c41698c92f2465356fe6263ecc56ed00515534139641a60200ac', 'src/round16_prefix_geo.py': '5f23b7edabc43a454b3bf870bb0832804a4686cc880dd5761491e46adadcc164', 'src/round17_ordered_geo.py': 'a04d0ef7b3765777dd411f5b9070143603f19a32e2986fd02efbf47abe3b834f', 'src/round18_structural_geo.py': '127afec1b700a5dca42b52d5c889bbc569270c181515695959b65976c9d7a86c', 'src/round19_unit_geo.py': 'fac832e6c81871833a136d05759c7f0926fb749d2320a9c39b581e3986dc75cf', 'src/round20_slow_geo.py': '3b82ff8908413543d28a3e0f78e36d86f1a87d09516119bda874a51d5486570f', 'src/round21_fine_grid_geo.py': '4ef8a93ebfc5298050a0ddd77b965b8baec12f5054c13b8b7e0aeb3c17c4ca19', 'src/round22_huber_feasibility.py': 'f071614ce1526f54943928a3cce0469e4510b01fa0de8348b2dbb9c3c9e424e9', 'src/round2_catboost.py': '63f1ca60edc56cf6e5093ba68124b4c4ce39b0f01654db05f14492a95cdabd13', 'src/round2_neighborhood.py': 'cae244c40e6616d6c72f3aff7c3423a50a5bfbfbf35df33a7d044186eb491ae9', 'src/round4_time_normalized.py': 'b4992f742aeb4134388c22b9d6e1ca578deafb31269b77b3ccdf85d2f201160a', 'src/round6_geo.py': 'df1454829b919e2a10ac264dd4e3f597367187174607a6d32abb20f40ef060a0', 'src/train_catboost.py': '852719f41cb739814ea6e7fc4b97d21be1a231d439d1c9aadebd343898ea590f'}
RESEARCH_FILES = ('outputs/round22_research/huber_candidate.cbm', 'outputs/round22_research/huber_feasibility.json', 'outputs/round22_research/huber_feasibility.log', 'outputs/round22_research/huber_feasibility.sha256', 'outputs/round22_research/mae_control.cbm', 'outputs/round22_research/method_sources.json', 'outputs/round22_research/method_sources.sha256', 'reports/round21_plan.md', 'reports/round22_method_review.md', 'runs/v1/models/final_config.json', 'src/prepare_data.py', 'src/round12_log_geo.py', 'src/round13_city_alias_geo.py', 'src/round14_geo_raw_lgb.py', 'src/round15_shallow_geo.py', 'src/round16_prefix_geo.py', 'src/round17_ordered_geo.py', 'src/round18_structural_geo.py', 'src/round19_unit_geo.py', 'src/round20_slow_geo.py', 'src/round21_fine_grid_geo.py', 'src/round22_huber_feasibility.py', 'src/round2_catboost.py', 'src/round2_neighborhood.py', 'src/round4_time_normalized.py', 'src/round6_geo.py', 'src/train_catboost.py')
HELPER_SHA = '3fddeb2b9c0d191dfae31538060c5713dbb3728ff1910c1cfe2b06777f0f1808'
SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA = '52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA = 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
THRESHOLDS = dict(e6.THRESHOLDS)
FORMULAS = {
    CONTROL: '.375*old_raw_cat+.375*geo_raw_cat+.25*old_unit_lgb; then v2 trend once',
    CANDIDATE: '.375*old_raw_cat+.375*huber_geo_catboost_TWD+.25*old_unit_lgb; then v2 trend once',
}
MODEL_FILES = {'model.cbm', 'feature_state.json', 'fit_indices.npy', 'query_indices.npy',
               'predictions.npy', 'raw_z.npy'}
NATIVE_CAPACITY_FIELDS = ('actual_model_depth', 'actual_boosting_type', 'actual_grow_policy',
                         'actual_trees', 'maximum_tree_leaf_count', 'tree_leaf_counts_sha256',
                         'actual_learning_rate', 'actual_loss_function', 'actual_eval_metric',
                         'actual_leaf_estimation_method', 'actual_leaf_estimation_iterations',
                         'actual_boost_from_average')
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
    paths += [geo/n for n in ['metadata.json','predictions.npy','fit_indices.npy','query_indices.npy']]
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
        require(spec[label+'_indices_content_sha256']==value_sha(array,'<i8'),'R22 partition index differs')
    base=c6.baseline_folder(stage);old=c6.OUT/'models'/stage/'geography_catboost'
    e6.verify_indices(base,fi,qi,'check_indices.npy');e6.verify_indices(old,fi,qi,'query_indices.npy')
    return data,X,y,Xq,yq,fi,qi

def publish(staging,destination):
    require(staging.resolve().parent==destination.resolve().parent and staging.resolve().is_relative_to(OUT.resolve()),'Unsafe R22 publication path')
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
            and baseline.get('loss_function') == 'MAE' and baseline.get('eval_metric') == 'MAE'
            and parameters == dict(baseline, loss_function='Huber:delta=1'))


def check_native_objective(native):
    for key, expected in NATIVE_OBJECTIVE_CONTRACT.items():
        value = native['actual_' + key]
        require(type(value) is type(expected) and value == expected,
                'Native objective/default differs: ' + key)



def research_bindings():
    require(bool(RESEARCH_EXPECTED_SHA), 'R22 synthetic compatibility research bindings are pending; plan prohibited')
    mapping = {name: sha(checked_path(name)) for name in RESEARCH_FILES}
    require(mapping == RESEARCH_EXPECTED_SHA,
            'Fixed synthetic Huber compatibility research bindings changed')
    return mapping


def plan_create():
    require(bool(RESEARCH_EXPECTED_SHA), 'R22 synthetic compatibility research bindings are pending; plan prohibited')
    require_no_staging()
    path = OUT / 'candidate_plan.json'
    require(not path.exists() and not path.with_suffix('.sha256').exists(), 'R22 plan exists or is partial; never overwrite')
    require(sha(Path(component.__file__)) == HELPER_SHA, 'Reviewed fixed Huber helper changed')
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
        'round': 22, 'created_at_utc': now(), 'candidate': CANDIDATE, 'control': CONTROL,
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'parameters': component.parameters(),
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
        'baseline_parameters': component.base.parameters(), 'changed_parameter_names': CHANGED_PARAMETERS,
        'removed_feature_names': REMOVED_FEATURES, 'added_feature_names': ADDED_FEATURES,
        'baseline_feature_count': 95, 'original95_preserved': True,
        'capacity_constraint': {'depth': 8, 'max_leaves_per_tree': 256, 'grow_policy': 'SymmetricTree', 'boosting_type': 'Plain'},
        'native_learning_rate_contract': {'expected': 0.065, 'rtol': 1e-7, 'atol': 0.0},
        'native_objective_contract': NATIVE_OBJECTIVE_CONTRACT,
        'implicit_native_policy_changes': IMPLICIT_NATIVE_POLICY_CHANGES,
        'sample_weight': 'uniform', 'query_labels_used_for_training': False,
        'shared_loader_deserializes_query_labels': True, 'query_labels_discarded_by_fit_caller': True,
        'geographic_feature_state_uses_query_or_y': False,
        'model_hyperparameter_change': {'loss_function': {'from': 'MAE', 'to': 'Huber:delta=1'}}, 'original_features_and_scoring_keys_preserved': True,
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
    require(bool(RESEARCH_EXPECTED_SHA), 'R22 synthetic compatibility research bindings are pending; plan prohibited')
    value = locked(OUT / 'candidate_plan.json')
    require(value['round'] == 22 and value['candidate'] == CANDIDATE and value['control'] == CONTROL,
            'Wrong R22 plan')
    require(value['formulas'] == FORMULAS and value['thresholds'] == THRESHOLDS
            and value['parameters'] == component.parameters(), 'Fixed configuration changed')
    require(value['baseline_parameters'] == component.base.parameters()
            and value['changed_parameter_names'] == CHANGED_PARAMETERS
            and value['removed_feature_names'] == REMOVED_FEATURES
            and value['added_feature_names'] == ADDED_FEATURES
            and value['baseline_feature_count'] == 95 and value['original95_preserved'] is True
            and schedule_is_fixed(value['parameters'], value['baseline_parameters'])
            and value['capacity_constraint'] == {'depth': 8, 'max_leaves_per_tree': 256, 'grow_policy': 'SymmetricTree', 'boosting_type': 'Plain'},
            'Fixed Huber schedule changed')
    require(value['native_learning_rate_contract'] == {'expected': 0.065, 'rtol': 1e-7, 'atol': 0.0},
            'Native learning rate contract changed')
    require(value['native_objective_contract'] == NATIVE_OBJECTIVE_CONTRACT
            and value['implicit_native_policy_changes'] == IMPLICIT_NATIVE_POLICY_CHANGES, 'Objective/default contract changed')
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
            and value['model_hyperparameter_change'] == {'loss_function': {'from': 'MAE', 'to': 'Huber:delta=1'}}
            and value['weight_argument_passed_to_pool'] is False
            and value['original_features_and_scoring_keys_preserved'] is True, 'Data-use declarations differ')
    require(all(value['dependency_sha256'].get(k) == v for k, v in value['research_dependency_sha256'].items()),
            'Research dependencies are outside frozen input map')
    c6.verify_hashes(value['dependency_sha256'])
    require(sha(ROOT / 'runs/versions.json') == value['registry_sha256_at_plan'], 'Preserved registry changed')
    return value


def validate_freeze(frozen, development, specification):
    require(frozen['round'] == 22 and frozen['candidate'] == CANDIDATE, 'Wrong development freeze')
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
    """Keep eleven native fields exact; allow only the fixed LR storage tolerance."""
    expected_keys = set(NATIVE_CAPACITY_FIELDS)
    require(isinstance(fitted, dict) and isinstance(reloaded, dict)
            and set(fitted) == set(reloaded) == expected_keys,
            'Fitted/reloaded native capacity field inventory differs')
    check_native_objective(fitted)
    check_native_objective(reloaded)
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


def validate_model_contract(meta, specification, expected):
    require(meta['round'] == 22 and meta['component'] == component.COMPONENT
            and meta['family'] == 'catboost', 'Wrong model component')
    check_native_objective(meta)
    require(meta['native_objective_contract'] == specification['native_objective_contract'] == NATIVE_OBJECTIVE_CONTRACT, 'Model objective contract differs')
    require(meta['baseline_parameters'] == specification['baseline_parameters']
            and meta['changed_parameter_names'] == CHANGED_PARAMETERS
            and meta['removed_feature_names'] == REMOVED_FEATURES
            and meta['added_feature_names'] == specification['added_feature_names'] == ADDED_FEATURES
            and meta['baseline_feature_count'] == 95 and meta['original95_preserved'] is True
            and schedule_is_fixed(meta['parameters'], meta['baseline_parameters'])
            and meta['actual_model_depth'] == 8 and meta['actual_grow_policy'] == 'SymmetricTree'
            and meta['actual_boosting_type'] == 'Plain'
            and meta['actual_trees'] == 2999 and 1 <= meta['maximum_tree_leaf_count'] <= 256,
            'Native model boosting/depth/leaf capacity differs')
    require(meta['parameters'] == specification['parameters']
            and meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23
            and meta['native_categorical_feature_count'] == 23
            and np.isclose(meta['actual_learning_rate'], 0.065, rtol=1e-7, atol=0),
            'Model parameters/features differ')
    require(meta['iterations'] == 2999 and meta['actual_iterations'] == 2999
            and meta['target_mode'] == component.TARGET_MODE and meta['sample_weight'] == 'uniform',
            'Model target/iteration/weight contract differs')
    require(meta['target_formula'] == component.TARGET_FORMULA
            and meta['inverse_formula'] == component.INVERSE_FORMULA, 'Model target scale differs')
    require(meta['feature_rules'] == component.FEATURE_RULES
            and meta['weight_argument_passed_to_pool'] is False, 'Feature/weight policy differs')
    for key in ('target_used_for_geographic_state', 'query_used_for_geographic_state',
                'query_labels_used', 'fit_uses_eval_set', 'early_stopping_used',
                'calibration_applied', 'extra_price_cap_applied', 'reference_or_trend_applied_here'):
        require(meta[key] is False, 'Forbidden model behavior: ' + key)
    require(meta['reload_all_query_rows_verified'] is True and meta['prediction_units'] == 'TWD'
            and meta['raw_z_diagnostics']['inverse_matches_component_all_rows'] is True,
            'Model replay/inverse contract differs')
    require(set(meta['artifact_sha256']) == MODEL_FILES, 'Incomplete or unexpected model artifact inventory')
    for key, value in expected.items():
        require(meta.get(key) == value, 'Model metadata alignment differs: ' + key)


def check_model_files(stage, specification):
    folder = OUT / 'models' / stage
    meta = locked(folder / 'metadata.json')
    expected = {
        'stage': stage, 'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'source_sha256': specification['source_sha256'],
        'outer_fit_rows': specification['split_contract'][stage]['fit_rows'],
        'query_rows': specification['split_contract'][stage]['query_rows'],
        'fit_indices_content_sha256': specification['split_contract'][stage]['fit_indices_content_sha256'],
        'query_indices_content_sha256': specification['split_contract'][stage]['query_indices_content_sha256'],
        'geo_manifest_sha256': sha(c6.OUT / 'geography' / stage / 'manifest.json'),
        'frozen_candidate_sha256': None if stage == 'development' else sha(OUT / 'frozen_candidate.json'),
    }
    validate_model_contract(meta, specification, expected)
    require(meta['feature_state_content_sha256'] == read(folder / 'feature_state.json')['state_content_sha256'],
            'Feature state content identity differs')
    require({p.name for p in folder.iterdir()} == MODEL_FILES | {'metadata.json', 'metadata.sha256'},
            'Model folder has incomplete or unexpected files')
    for name, digest in meta['artifact_sha256'].items():
        require(Path(name).name == name and sha(folder / name) == digest, 'Saved model artifact changed')
    return meta, {rel(p): sha(p) for p in folder.iterdir() if p.is_file()}




def fit(stage):
    require_no_staging()
    specification = authorize(stage)
    destination = OUT / 'models' / stage
    require(not destination.exists(), 'Model already exists; never retrain')
    data, X, y, Xq, yq, fi, qi = partition(stage)
    del yq  # The shared loader deserialized labels; the component never receives query labels.
    geo, geoq, _ = load_geographic_partition(stage, fi, qi)
    before = feature_hashes(X, Xq, geo, geoq)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + stage + '-' + uuid.uuid4().hex)
    staging.mkdir()
    started = time.monotonic()
    cats = data['categorical_columns']
    print(json.dumps({'event': 'fit_started', 'stage': stage, 'fit_rows': len(X),
                      'query_rows': len(Xq), 'features': 95, 'parameters': specification['parameters']}), flush=True)
    model, state, metadata = component.fit_model(X, y, geo, cats)
    predictions = component.predict_model(model, state, Xq, geoq, cats)
    z = raw_predict(model, state, Xq, geoq, cats)
    diagnostics = raw_diagnostics(z, predictions)
    model.save_model(str(staging / 'model.cbm'))
    write(staging / 'feature_state.json', state)
    reloaded = load_model(staging / 'model.cbm')
    restored = read(staging / 'feature_state.json')
    replay = component.predict_model(reloaded, restored, Xq, geoq, cats)
    reloaded_native = component.check_model(reloaded, restored)
    fitted_native = {key: metadata[key] for key in NATIVE_CAPACITY_FIELDS}
    metadata.update(compare_reloaded_native_capacity(fitted_native, reloaded_native))
    require(np.array_equal(predictions, replay), 'Saved model full-query replay differs')
    require(int(reloaded.tree_count_) == 2999, 'Reloaded CatBoost has unexpected iteration count')
    require(before == feature_hashes(X, Xq, geo, geoq), 'Model operations changed X/geo values')
    for name, values in [('fit_indices', fi), ('query_indices', qi), ('predictions', predictions),
                         ('raw_z', z)]:
        np.save(staging / (name + '.npy'), values, allow_pickle=False)
    metadata['component_source_sha256'] = metadata['source_sha256']
    metadata.update({
        'round': 22, 'stage': stage, 'created_at_utc': now(),
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'source_sha256': specification['source_sha256'],
        'actual_iterations': int(reloaded.tree_count_), 'outer_fit_rows': len(X), 'query_rows': len(Xq),
        'fit_target_sha256': value_sha(y, '<f8'), **before,
        'fit_indices_content_sha256': value_sha(fi, '<i8'),
        'query_indices_content_sha256': value_sha(qi, '<i8'),
        'geo_manifest_sha256': sha(c6.OUT / 'geography' / stage / 'manifest.json'),
        'artifact_sha256': {p.name: sha(p) for p in staging.iterdir() if p.is_file()},
        'reload_all_query_rows_verified': True, 'reload_query_rows': len(Xq),
        'reload_exact_equality': True, 'raw_z_diagnostics': diagnostics,
        'shared_loader_deserialized_query_labels': True, 'query_labels_discarded_by_fit_caller': True,
        'query_labels_used': False, 'prediction_units': 'TWD', 'direct_component_score_omitted': True,
        'runner_elapsed_seconds': time.monotonic() - started,
        'frozen_candidate_sha256': None if stage == 'development' else sha(OUT / 'frozen_candidate.json'),
    })
    validate_model_contract(metadata, specification, before)
    locked_write(staging / 'metadata.json', metadata)
    authorize(stage)
    publish(staging, destination)
    print(json.dumps({'event': 'fit_completed', 'stage': stage,
                      'elapsed_seconds': time.monotonic() - started,
                      'raw_negative_count': diagnostics['negative_count']}), flush=True)


def model_check(stage, X, y, Xq, fi, qi, geo, geoq, cats, specification):
    meta, inputs = check_model_files(stage, specification)
    before = feature_hashes(X, Xq, geo, geoq)
    validate_model_contract(meta, specification, before)
    require(meta['fit_target_sha256'] == meta['fit_target_values_sha256']
            == meta['fit_original_target_values_sha256'] == value_sha(y, '<f8'),
            'Original fit target identity differs')
    require(meta['fit_transformed_target_values_sha256']
            == value_sha(component.transform_target(y), '<f8'), 'Scaled raw fit target identity differs')
    folder = OUT / 'models' / stage
    e6.verify_indices(folder, fi, qi, 'query_indices.npy')
    saved = e6.vector(folder / 'predictions.npy', len(qi))
    z = np.load(folder / 'raw_z.npy', allow_pickle=False)
    require(raw_diagnostics(z, saved) == meta['raw_z_diagnostics'], 'Saved raw-output diagnostics differ')
    state = read(folder / 'feature_state.json')
    require(state == component.fit_feature_state(X, geo, cats), 'Saved state differs from exact fit-only state')
    model = load_model(folder / 'model.cbm')
    require(all(meta.get(k) == v for k, v in component.check_model(model, state).items()),
            'Saved native capacity diagnostics differ')
    require(int(model.tree_count_) == 2999, 'Reloaded model iteration count differs')
    replay = component.predict_model(model, state, Xq, geoq, cats)
    require(np.array_equal(saved, replay), 'Full-query saved-model replay differs')
    require(np.array_equal(z, raw_predict(model, state, Xq, geoq, cats)), 'Full-query raw model replay differs')
    require(before == feature_hashes(X, Xq, geo, geoq), 'Replay changed X/geo values')
    return saved, inputs


def validate_result_contract(stage, value, specification):
    require(value['round'] == 22 and value['stage'] == stage
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
        'created_at_utc': now(), 'round': 22, 'candidate': CANDIDATE,
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'result_sha256': sha(OUT / 'evaluation/development/results.json'),
        'passed': result['decision']['passed'], 'selected_candidate': result['decision']['selected_candidate'],
        'development_artifact_sha256': artifacts, 'source_sha256': specification['source_sha256'],
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'parameters': component.parameters(),
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
        'round': 22, 'stage': stage, 'created_at_utc': now(),
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
        'created_at_utc': now(), 'round': 22, 'plan_sha256': sha(OUT / 'candidate_plan.json'),
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
    """In-memory synthetic gates, raw outputs and inverse, and FakePool checks."""
    import copy
    from unittest.mock import patch
    checks = []
    component_checks = component.self_check()

    def run(name, truth):
        require(bool(truth), 'Synthetic check failed: ' + name)
        checks.append(name)

    def rejects(name, operation):
        try:
            operation()
        except (RuntimeError, ValueError, AssertionError, KeyError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)

    schedule_base = {'iterations': 2999, 'learning_rate': 0.065, 'boosting_type': 'Plain', 'depth': 8,
                     'loss_function': 'MAE', 'eval_metric': 'MAE'}
    schedule_new = dict(schedule_base, loss_function='Huber:delta=1')
    run('passed_parameter_difference_is_only_Huber_loss', schedule_is_fixed(schedule_new, schedule_base))
    for key, value in [('iterations', 5998), ('learning_rate', .0325), ('depth', 9), ('boosting_type', 'Ordered'),
                       ('loss_function', 'Huber:delta=2'), ('eval_metric', 'RMSE')]:
        run('schedule_rejects_' + key, not schedule_is_fixed(dict(schedule_new, **{key: value}), schedule_base))
    run('unexpected_parameter_rejected', not schedule_is_fixed(dict(schedule_new, extra=True), schedule_base))
    fitted_native = {
        'actual_model_depth': 8, 'actual_boosting_type': 'Plain', 'actual_grow_policy': 'SymmetricTree',
        'actual_trees': 2999, 'maximum_tree_leaf_count': 256, 'tree_leaf_counts_sha256': 'fixed_leaf_digest',
        'actual_learning_rate': float(np.float32(.065)),
        **{'actual_' + k: v for k, v in NATIVE_OBJECTIVE_CONTRACT.items()},
    }
    reloaded_native = dict(fitted_native, actual_learning_rate=.06499999762)
    correction = compare_reloaded_native_capacity(fitted_native, reloaded_native)
    run('native_LR_storage_representations_accepted',
        fitted_native['actual_learning_rate'] != reloaded_native['actual_learning_rate']
        and correction['actual_learning_rate'] == reloaded_native['actual_learning_rate']
        and correction['fitted_actual_learning_rate'] == fitted_native['actual_learning_rate'])
    run('eleven_non_LR_native_fields_exact', all(correction[k] == fitted_native[k] for k in NATIVE_CAPACITY_FIELDS if k != 'actual_learning_rate'))
    for key, value in [('actual_loss_function','MAE'),('actual_loss_function','Huber:delta=2'),
                       ('actual_eval_metric','Huber:delta=1'),('actual_leaf_estimation_method','Exact'),
                       ('actual_leaf_estimation_iterations',2),('actual_boost_from_average',True),
                       ('actual_boost_from_average',0),('actual_learning_rate',.0325),
                       ('actual_learning_rate',float('nan')),('actual_trees',5998),
                       ('tree_leaf_counts_sha256','different')]:
        rejects('native_saved_rejects_'+key+'_'+str(value),lambda key=key,value=value:compare_reloaded_native_capacity(fitted_native,dict(reloaded_native,**{key:value})))
    rejects('both_wrong_objectives_rejected',lambda:compare_reloaded_native_capacity(dict(fitted_native,actual_loss_function='MAE'),dict(reloaded_native,actual_loss_function='MAE')))
    rejects('missing_native_field_rejected',lambda:compare_reloaded_native_capacity({k:v for k,v in fitted_native.items() if k!='actual_trees'},reloaded_native))
    rejects('extra_native_field_rejected',lambda:compare_reloaded_native_capacity(fitted_native,dict(reloaded_native,extra=True)))
    run('original95_23_no_feature_changes', component.FEATURE_RULES['feature_count']==95
        and component.FEATURE_RULES['categorical_feature_count']==23
        and component.FEATURE_RULES['added_feature_names']==[]
        and component.FEATURE_RULES['removed_feature_names']==[]
        and component.FEATURE_RULES['changed_parameter_names']==['loss_function'])
    with patch.dict(globals(), {'RESEARCH_EXPECTED_SHA': {}, 'RESEARCH_FILES': ()}):
        rejects('pending_research_bindings_reject_plan_before_any_input_IO', plan_create)
    base = {'overall_mae': 100., 'weighted_mae': 100.,
            'strata': {'seen': {'rows': 500, 'mae': 100.}, 'new': {'rows': 499, 'mae': 100.},
                       'missing': {'rows': 0, 'mae': None}},
            'forecast_quarters': {q: {'rows': 10, 'mae': 100.} for q in ('Q1', 'Q2', 'Q3', 'Q4')}}

    def metrics(overall, weighted=None):
        value = copy.deepcopy(base)
        value['overall_mae'], value['weighted_mae'] = overall, overall if weighted is None else weighted
        return {CONTROL: copy.deepcopy(base), CANDIDATE: value}

    run('development_exact_dual_1pct_passes', decide('development', metrics(99))['passed'])
    run('development_below_1pct_rejected', not decide('development', metrics(99.00001))['passed'])
    run('weighted_1pct_also_required', not decide('development', metrics(98, 100))['passed'])
    m = metrics(98); m[CANDIDATE]['strata']['seen']['mae'] = 105.00001
    run('500_row_stratum_deterioration_gate', not decide('development', m)['passed'])
    m = metrics(98); m[CANDIDATE]['strata']['new']['mae'] = 1000.
    run('499_row_stratum_not_new_gate', decide('development', m)['passed'])
    m = metrics(98); m[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 105.00001
    run('nonempty_quarter_deterioration_gate', not decide('development', m)['passed'])
    m = metrics(99); m[CANDIDATE]['forecast_quarters']['Q1']['mae'] = 99.
    run('confirmation_one_quarter_insufficient', not decide('confirmation', m)['passed'])
    m[CANDIDATE]['forecast_quarters']['Q2']['mae'] = 99.
    run('confirmation_two_quarters_pass', decide('confirmation', m)['passed'])
    m[CANDIDATE]['weighted_mae'] = 100.00001
    run('confirmation_weighted_non_deterioration_required', not decide('confirmation', m)['passed'])
    run('auxiliary_exact_3pct_passes', decide('auxiliary', metrics(103))['passed'])
    run('auxiliary_over_3pct_rejected', not decide('auxiliary', metrics(103.00001))['passed'])
    rejects('extra_candidate_rejected', lambda: decide('development', {**metrics(98), 'extra': base}))
    historical = {'outputs/round6/candidate_plan.json': 'original_plan',
                  'outputs/round6/candidate_plan.sha256': 'original_receipt'}
    validate_original_plan_binding(historical, 'original_plan', 'original_receipt', 'original_plan')
    run('original_R6_plan_and_receipt_match_released_authority', True)
    rejects('replacement_R6_plan_or_source_authority_rejected',
            lambda: validate_original_plan_binding(historical, 'replacement_plan', 'original_receipt', 'replacement_plan'))
    rejects('replacement_R6_receipt_rejected',
            lambda: validate_original_plan_binding(historical, 'original_plan', 'replacement_receipt', 'original_plan'))
    rejects('receipt_pointing_to_other_plan_rejected',
            lambda: validate_original_plan_binding(historical, 'original_plan', 'original_receipt', 'different_plan'))
    validate_original_plan_binding({'outputs/round6/candidate_plan.json': 'original_plan'},
                                   'original_plan', 'receipt_unlisted_in_snapshot', 'original_plan')
    run('unlisted_receipt_still_binds_exact_released_plan_content', True)
    calls = []; query = [object(), object()]
    def fake_reference(reference, X, values, variant):
        require(reference == 'fit_only' and X is query and variant == 'trend', 'Reference call differs')
        calls.append(values.copy())
        return values * 1.1
    prices = blend([1, 2], [3, 4], [5, 6], 'fit_only', query, fake_reference)
    run('fixed_375_375_25_and_exactly_one_trend', len(calls) == 1 and np.allclose(
        prices, 1.1 * (.375*np.array([1, 2]) + .375*np.array([3, 4]) + .25*np.array([5, 6]))))
    selected = {'passed': True, 'later_stages_allowed': True,
                'selected_candidate': CANDIDATE, 'plan_sha256': 'fixed'}
    check_later_gate('confirmation', selected, 'fixed')
    check_later_gate('auxiliary', selected, 'fixed')
    run('own_passed_freeze_authorizes_both_later_stages', True)
    for key, value in [('passed', False), ('later_stages_allowed', False),
                       ('selected_candidate', CONTROL), ('plan_sha256', 'different')]:
        bad = dict(selected); bad[key] = value
        rejects('later_gate_rejects_' + key, lambda: check_later_gate('confirmation', bad, 'fixed'))
    rejects('final_fit_not_implemented', lambda: check_later_gate('final', selected, 'fixed'))
    reject = {'decision': {'passed': False}}
    passed = {'decision': {'passed': True}}
    run('development_failure_stops_without_later_results',
        final_outcome({'development': reject}) == (False, 'development_price_gates_failed'))
    rejects('missing_later_result_is_error_not_rejection', lambda: final_outcome({'development': passed}))
    run('three_passed_backtests_not_release',
        final_outcome({stage: passed for stage in STAGES}) == (True, 'all_fixed_backtests_passed'))
    later_failed = {stage: passed for stage in STAGES}; later_failed['auxiliary'] = reject
    run('later_gate_is_veto', final_outcome(later_failed) == (False, 'later_backtest_gates_failed'))
    z = np.array([-0.2, 0., 2.], dtype=np.float64)
    diagnostics = raw_diagnostics(z, component.inverse_transform(z))
    run('negative_raw_count_not_zero_price_count',
        diagnostics['negative_count'] == 1 and diagnostics['zero_count'] == 1
        and diagnostics['zero_price_count'] == 2 and diagnostics['positive_count'] == 1)
    rejects('wrong_inverse_rejected', lambda: raw_diagnostics(z, np.zeros(3)))
    rejects('nonfinite_raw_output_rejected', lambda: raw_diagnostics(np.array([np.nan]), np.array([0.])))
    rejects('overflow_raw_output_rejected', lambda: raw_diagnostics(np.array([np.finfo(np.float64).max]), np.array([0.])))
    mapping = {'src/fixed.py': 'source', 'indices.npy': 'index',
               **{name + '.pkl': name for name in ('train', 'valid', 'test', 'full')}}
    manifest = {
        'stage': 'development', 'snapshot_year': 2022,
        'geography_provenance_chain': {'verified': True},
        'feature_rules': component.base.FEATURE_RULES,
        'replication_timestamp': '2022-01-01T00:00:00Z',
        'forecast_origin_utc': '2022-08-10T16:00:00Z',
        'source_sha256': {'src/fixed.py': 'source'}, 'index_files_sha256': {'indices.npy': 'index'},
        'features': {name: {'path': name + '.pkl', 'sha256': name}
                     for name in ('train', 'valid', 'test', 'full')},
    }
    validate_geo_contract('development', manifest, mapping)
    run('fixed_year_geo_manifest_closure', True)
    for key, value in [('snapshot_year', 2023), ('replication_timestamp', '2023-01-01T00:00:00Z'),
                       ('source_sha256', {'src/fixed.py': 'changed'})]:
        bad = copy.deepcopy(manifest); bad[key] = value
        rejects('geo_rejects_' + key, lambda: validate_geo_contract('development', bad, mapping))
    bad = copy.deepcopy(manifest); bad['features']['train']['sha256'] = 'different'
    rejects('geo_rejects_frame_identity', lambda: validate_geo_contract('development', bad, mapping))
    with patch.object(Path, 'exists', return_value=True), patch.object(Path, 'iterdir',
            return_value=iter([Path('.development-interrupted')])):
        rejects('partial_staging_fails_closed_without_deletion', require_no_staging)
    fit_ast = next(n for n in ast.parse(Path(__file__).read_text(encoding='utf-8')).body
                   if isinstance(n, ast.FunctionDef) and n.name == 'fit')
    component_fit_calls = [n for n in ast.walk(fit_ast) if isinstance(n, ast.Call)
                           and isinstance(n.func, ast.Attribute) and n.func.attr == 'fit_model']
    run('fit_passes_no_query_target_to_component', len(component_fit_calls) == 1
        and [ast.unparse(arg) for arg in component_fit_calls[0].args] == ['X', 'y', 'geo', 'cats']
        and not component_fit_calls[0].keywords)
    run('source_does_not_import_domain_weights_or_POI',
        not any(name.endswith(('/round11_domain_weights.py', '/round9_geo_poi.py')) for name in sources()))
    proof = {
        'created_at_utc': now(), 'all_checks_passed': True, 'synthetic_checks': checks,
        'component_synthetic_checks': component_checks, 'source_sha256': sources(),
        'real_model_training_or_scoring': False, 'parameters': component.parameters(),
        'runtime': runtime_versions(), 'round': 22,
    }
    if write_output:
        locked_write(OUT / 'audit/preflight.json', proof)
    print(json.dumps(proof, ensure_ascii=False), flush=True)
    return proof


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

