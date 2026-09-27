"""R11: one prepared pure-X importance-weighted geographic CatBoost replacement.
This backtest driver cannot release a CSV, fit final, or modify old experiments.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, time, uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from catboost import CatBoostRegressor
import round6_common as c6
import round6_evaluate as e6
import round11_weighted_geo as component
from round6_prepare_geo import load_geographic_partition
from round6_geo import frame_sha
from round2_neighborhood import predict_reference
from round3_evaluate import score, target_weights, runtime_versions
ROOT = c6.ROOT
OUT = ROOT / 'outputs/round11'
STAGES = tuple(c6.SPLITS)
CANDIDATE = 'importance_geo_blend'
CONTROL = 'v3_control'
EXTRA_SOURCES = ('src/round11_domain_weights.py', 'src/round11_prepare_weights.py')
REQUIRED_DOCUMENTS = (
    'reports/round11_domain_plan.md',
    'reports/round11_pipeline_design.md',
    'reports/round11_covariate_shift_feasibility.md',
    'reports/round11_price_pipeline_review.md',
    'reports/round11_weight_result_review.md',
    'outputs/round11/weight_result_review.json',
    'outputs/round11/weight_result_review.sha256',
)
SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA = '52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA = 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
THRESHOLDS = dict(e6.THRESHOLDS)
FORMULAS = {
    CONTROL: '.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once',
    CANDIDATE: '.75*(.5*old_raw_cat+.5*importance_geo_raw_cat)+.25*old_unit_lgb; then v2 trend once',
}
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


def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def checked_path(name):
    path=(ROOT/Path(name)).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(),'Missing/unsafe dependency: '+str(name))
    return path


def load_model(path):
    model=CatBoostRegressor()
    model.load_model(str(path))
    return model


def check_later_gate(stage,frozen,plan_sha):
    require(stage in ('confirmation','auxiliary'),'Only declared later backtests may be authorized')
    require(frozen['passed'] is True and frozen['later_stages_allowed'] is True and frozen['selected_candidate']==CANDIDATE,'Development rejected; later stages prohibited')
    require(frozen['plan_sha256']==plan_sha,'Development selection belongs to another plan')


def publish(staging,destination):
    require(staging.resolve().parent==destination.resolve().parent and staging.resolve().is_relative_to(OUT.resolve()),'Unsafe R11 publication path')
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


def verify_domain_plan(specification=None):
    """Read hash-verified pure-X preparation authority, never labels or arrays."""
    import round11_domain_weights as domain
    from round11_prepare_weights import runtime as preparation_runtime
    path = OUT / 'domain_plan.json'
    value = locked(path)
    require(value['round'] == 11, 'Wrong domain plan round')
    require(value['runtime_versions'] == preparation_runtime(), 'Current domain runtime changed')
    require(value['config'] == domain.CONFIG and value['config_sha256'] == domain.content_sha(domain.CONFIG),
            'Fixed domain preparation configuration changed')
    require(value['reference_snapshot_sha256'] == SNAPSHOT_SHA, 'Domain plan uses another reference snapshot')
    c6.verify_hashes(value['source_sha256'])
    c6.verify_hashes(value['input_sha256'])
    require(set(EXTRA_SOURCES) <= set(value['source_sha256']), 'Domain preparation source closure incomplete')
    current = sources() if specification is None else specification['source_sha256']
    for name, digest in value['source_sha256'].items():
        require(current.get(name) == digest, 'Domain source absent/different in price closure: ' + name)
    for stage, dates in c6.SPLITS.items():
        require(value['splits'][stage] == list(dates), 'Domain/price window definitions differ')
    if specification is not None:
        require(sha(path) == specification['domain_plan_sha256'], 'Domain plan changed')
        require(value['config'] == specification['domain_config']
                and value['config_sha256'] == specification['domain_config_sha256'], 'Domain configuration differs')
        require(value['runtime_versions'] == specification['domain_runtime_versions'], 'Domain runtime differs')
    inputs = {rel(path): sha(path), rel(path.with_suffix('.sha256')): sha(path.with_suffix('.sha256'))}
    inputs.update(value['source_sha256']); inputs.update(value['input_sha256'])
    return value, inputs


def validate_prepared_contract(manifest, specification, stage, require_pass=True):
    require(stage in STAGES, 'Only fixed backtest preparation is accepted')
    require(manifest['round'] == 11 and manifest['stage'] == stage, 'Wrong prepared weights identity')
    require(manifest['domain_plan_sha256'] == specification['domain_plan_sha256'], 'Prepared domain plan differs')
    require(manifest['runtime_versions'] == specification['domain_runtime_versions'], 'Prepared domain runtime differs')
    require(manifest['config'] == specification['domain_config'], 'Prepared domain configuration differs')
    require(manifest['query_X_used_for_weights'] is True and manifest['query_y_or_price_cache_used'] is False
            and manifest['original_features_mutated'] is False, 'Prepared data-use contract differs')
    import round11_domain_weights as domain
    computed = domain.feasibility_checks(manifest['weight_summary'], manifest['coverage'])
    require(manifest['feasibility_checks'] == computed
            and manifest['feasibility_passed'] is all(computed.values()), 'Pure-X feasibility evidence is inconsistent')
    if require_pass:
        require(manifest['feasibility_passed'] is True, 'Pure-X feasibility did not pass')
    split = specification['split_contract'][stage]
    for key in ['fit_indices_content_sha256', 'query_indices_content_sha256']:
        require(manifest[key] == split[key], 'Prepared canonical row order differs: ' + key)
    for name, digest in manifest['source_sha256'].items():
        require(specification['source_sha256'].get(name) == digest, 'Unfrozen preparation source: ' + name)
    require(set(EXTRA_SOURCES) <= set(manifest['source_sha256']), 'Prepared source closure incomplete')


def verify_weight_manifest(stage, specification, require_pass=True):
    """Expand only the authoritative source/input/artifact maps, including folds."""
    domain_authority, inputs = verify_domain_plan(specification)
    folder = OUT / 'weights' / stage
    path = folder / 'manifest.json'
    manifest = locked(path)
    validate_prepared_contract(manifest, specification, stage, require_pass=require_pass)
    require(manifest['source_sha256'] == domain_authority['source_sha256']
            and manifest['input_sha256'] == domain_authority['input_sha256'],
            'Preparation source/input inventory differs from frozen domain authority')
    expected_files = ['fit_folds.npy', 'query_folds.npy', 'fit_oof_probability.npy',
                      'query_oof_probability.npy', 'fit_raw_log_ratio.npy', 'query_raw_log_ratio.npy',
                      'fit_clipped_ratio.npy', 'fit_weights.npy']
    expected_files += ['fold_' + str(fold) + '/' + name for fold in range(5)
                       for name in ['classifier.joblib', 'preprocessor.json']]
    require(set(manifest['artifact_sha256']) == {rel(folder / name) for name in expected_files},
            'Prepared fold/weight artifact inventory is incomplete or unexpected')
    inputs.update({rel(path): sha(path), rel(path.with_suffix('.sha256')): sha(path.with_suffix('.sha256'))})
    known = dict(specification['source_sha256'])
    known.update(specification['dependency_sha256'])
    known.update(specification['stage_dependency_sha256'][stage])
    for field in ['source_sha256', 'input_sha256', 'artifact_sha256']:
        mapping = manifest[field]
        require(isinstance(mapping, dict) and mapping, 'Empty prepared manifest map: ' + field)
        for name, digest in mapping.items():
            artifact = checked_path(name)
            if field == 'artifact_sha256':
                require(artifact.is_relative_to(folder.resolve()) and artifact != path.resolve()
                        and artifact != path.with_suffix('.sha256').resolve(), 'Prepared artifact outside its stage')
            if name in known:
                require(known[name] == digest, 'Prepared identity differs from frozen input: ' + name)
            if name.startswith('src/') and name.endswith('.py'):
                require(specification['source_sha256'].get(name) == digest, 'Unfrozen prepared Python dependency')
            inputs[name] = digest
    c6.verify_hashes(inputs)
    return manifest, inputs


def validate_weight_identity(contract, stage, X, Xq, fi, qi, weights, specification, manifest_sha):
    for label, indices, rows in [('fit', fi, len(X)), ('query', qi, len(Xq))]:
        array = np.asarray(indices)
        require(array.dtype == np.dtype('int64') and array.shape == (rows,)
                and len(np.unique(array)) == rows, 'Invalid canonical ' + label + ' indices')
    require(not np.intersect1d(fi, qi).size, 'Canonical fit/query rows overlap')
    expected = {
        'phase': stage, 'fit_X_sha256': frame_sha(X), 'query_X_sha256': frame_sha(Xq),
        'fit_indices_content_sha256': value_sha(fi, '<i8'),
        'query_indices_content_sha256': value_sha(qi, '<i8'),
        'weights_values_sha256': component.weights_values_sha256(weights),
        'domain_plan_sha256': specification['domain_plan_sha256'],
        'preparation_manifest_sha256': manifest_sha,
        'query_X_used_for_weights': True, 'query_labels_used_for_weights': False,
    }
    require(contract == expected, 'Prepared weights belong to another full fit/query batch')
    component.validate_weights(X, weights, contract)
    return expected


def load_weights(stage, X, Xq, fi, qi, specification):
    from round11_prepare_weights import load_prepared
    manifest, inputs = verify_weight_manifest(stage, specification)
    weights, contract, loaded = load_prepared(stage, X, Xq, fi, qi)
    require(loaded == manifest, 'Preparation changed while loading weights')
    manifest_sha = sha(OUT / 'weights' / stage / 'manifest.json')
    validate_weight_identity(contract, stage, X, Xq, fi, qi, weights, specification, manifest_sha)
    expected = {
        'fit_X_sha256': contract['fit_X_sha256'], 'query_X_sha256': contract['query_X_sha256'],
        'fit_indices_content_sha256': contract['fit_indices_content_sha256'],
        'query_indices_content_sha256': contract['query_indices_content_sha256'],
        'fit_weight_values_sha256': contract['weights_values_sha256'],
    }
    for key, value in expected.items():
        require(manifest[key] == value, 'Prepared weight/row identity mismatch: ' + key)
    c6.verify_hashes(inputs)
    return weights, contract, manifest, inputs


def validate_model_contract(meta, specification, expected):
    require(meta['round'] == 11 and meta['component'] == component.COMPONENT, 'Wrong weighted model identity')
    require(meta['parameters'] == specification['parameters'] and meta['feature_count'] == 95
            and meta['categorical_feature_count'] == 23, 'Model parameters/features differ')
    require(meta['iterations'] == 2999 and meta['actual_iterations'] == 2999
            and meta['target_mode'] == 'raw' and meta['sample_weight'] == 'importance_clip_then_mean_one',
            'Model target/iterations/importance-weight contract differs')
    require(meta['feature_rules'] == component.FEATURE_RULES
            and meta['target_formula'] == component.FEATURE_RULES['fit_target']
            and meta['prediction_formula'] == component.FEATURE_RULES['prediction'], 'Model price scale/rules differ')
    require(meta['query_X_used_for_training_weights'] is True and meta['query_labels_used'] is False,
            'Transductive weighting or label-use declaration differs')
    require(meta['target_used_for_geographic_state'] is False and meta['query_used_for_geographic_state'] is False
            and meta['reference_features_changed'] is False and meta['reference_or_trend_applied_here'] is False,
            'Geographic features/reference changed')
    require(meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False,
            'Fixed full-iteration fitting contract differs')
    for key, value in expected.items():
        require(meta.get(key) == value, 'Model metadata alignment differs: ' + key)


def plan_create():
    require(not (OUT / 'candidate_plan.json').exists(), 'R11 plan exists; never overwrite')
    snapshot_path = ROOT / 'runs/v3/snapshot_manifest.json'
    require(sha(snapshot_path) == SNAPSHOT_SHA, 'Released v3 snapshot changed')
    snapshot = locked(snapshot_path)
    known = {k: v['original_sha256'] for k, v in snapshot['copied_files'].items()}
    known.update({v['original_path']: v['sha256'] for v in snapshot['external_dependencies']})
    old = c6.plan(); deps = dict(old['data_sha256'])
    required = [ROOT / 'outputs/submission_v3.csv', ROOT / 'outputs/submission_v2.csv',
                snapshot_path, snapshot_path.with_suffix('.sha256'),
                c6.OUT / 'candidate_plan.json', c6.OUT / 'candidate_plan.sha256',
                ROOT / 'runs/v1/models/final_config.json', OUT / 'audit/preflight.json']
    required += [ROOT / name for name in REQUIRED_DOCUMENTS]
    for path in required:
        deps[rel(path)] = sha(path)
    require(deps['outputs/submission_v3.csv'] == V3_SHA and deps['outputs/submission_v2.csv'] == V2_SHA,
            'Preserved submissions changed')
    preflight = read(OUT / 'audit/preflight.json')
    require(preflight['all_checks_passed'] is True and preflight['real_model_training_or_scoring'] is False,
            'Synthetic preflight did not pass')
    require(preflight['source_sha256'] == sources(), 'Source changed after preflight')
    stages = {}
    for stage in STAGES:
        mapping = {}
        for path in stage_paths(stage):
            name = rel(path); digest = sha(path)
            require(known.get(name) == digest, 'Stage dependency differs from v3 snapshot: ' + name)
            mapping[name] = digest
        stages[stage] = mapping
    final_manifest = c6.OUT / 'geography/final/manifest.json'; final_geo = locked(final_manifest)
    final_paths = [final_manifest, final_manifest.with_suffix('.sha256'),
                   c6.OUT / 'release/release.json', c6.OUT / 'release/release.sha256']
    final_paths += [ROOT / item['path'] for item in final_geo['features'].values()]
    final_paths += [ROOT / p for field in ['source_sha256', 'index_files_sha256'] for p in final_geo[field]]
    final_dependencies = {}
    for path in sorted(set(final_paths)):
        name = rel(path); digest = sha(path)
        require(known.get(name) == digest, 'Final geography differs from v3 snapshot: ' + name)
        final_dependencies[name] = digest
    domain, domain_inputs = verify_domain_plan()
    deps.update(domain_inputs)
    value = {
        'round': 11, 'created_at_utc': now(), 'candidate': CANDIDATE, 'control': CONTROL,
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'parameters': component.parameters(),
        'feature_count': 95, 'categorical_count': 23, 'feature_rules': component.FEATURE_RULES,
        'domain_plan_sha256': sha(OUT / 'domain_plan.json'),
        'domain_config': domain['config'], 'domain_config_sha256': domain['config_sha256'],
        'domain_runtime_versions': domain['runtime_versions'],
        'source_sha256': sources(), 'dependency_sha256': deps, 'stage_dependency_sha256': stages,
        'final_dependency_sha256': final_dependencies, 'final_contract': old['final_contract'],
        'splits': old['splits'], 'split_contract': old['split_contract'],
        'target_group_weights': old['target_group_weights'], 'quarter_definition': old['quarter_definition'],
        'old_plan_sha256': sha(c6.OUT / 'candidate_plan.json'), 'v3_snapshot_sha256': SNAPSHOT_SHA,
        'single_development_candidate': True, 'tuning_or_weight_search': False,
        'query_X_used_for_training_weights': True, 'query_labels_used_for_training': False,
        'geographic_feature_state_uses_query': False,
        'training_importance_weights_are_not_evaluation_group_weights': True,
        'query_batch_identity_is_part_of_preparation_contract': True,
        'public_score_used_for_selection': False, 'historically_untouched': False,
        'external_property_answers_used': False, 'fit_uses_eval_set': False,
        'early_stopping_used': False, 'runtime': runtime_versions(),
        'registry_sha256_at_plan': sha(ROOT / 'runs/versions.json'),
        'final_fit_implemented_in_this_driver': False,
        'later_weight_preparation_requires_this_round_development_pass': True,
    }
    _, prepared_inputs = verify_weight_manifest('development', value)
    deps.update(prepared_inputs)
    c6.verify_hashes(deps)
    require(value['source_sha256'] == preflight['source_sha256'] == sources(),
            'Sources changed after preflight or while freezing price plan')
    locked_write(OUT / 'candidate_plan.json', value)
    print(json.dumps({'status': 'plan_frozen', 'plan_sha256': sha(OUT / 'candidate_plan.json'),
                      'candidate': CANDIDATE}), flush=True)


def plan():
    value = locked(OUT / 'candidate_plan.json')
    require(value['round'] == 11 and value['candidate'] == CANDIDATE and value['control'] == CONTROL,
            'Wrong R11 price plan')
    require(value['formulas'] == FORMULAS and value['thresholds'] == THRESHOLDS
            and value['parameters'] == component.parameters(), 'Fixed price configuration changed')
    require(value['feature_rules'] == component.FEATURE_RULES and value['feature_count'] == 95
            and value['categorical_count'] == 23, 'Original 95-feature contract changed')
    require(value['source_sha256'] == sources(), 'Frozen price/source closure changed')
    require(value['splits'] == {k: list(v) for k, v in c6.SPLITS.items()}, 'Split definitions changed')
    require(value['runtime'] == runtime_versions(), 'Price-model runtime changed')
    require(value['query_X_used_for_training_weights'] is True
            and value['query_labels_used_for_training'] is False
            and value['geographic_feature_state_uses_query'] is False, 'Data-use declarations differ')
    c6.verify_hashes(value['dependency_sha256'])
    require(sha(ROOT / 'runs/versions.json') == value['registry_sha256_at_plan'], 'Preserved registry changed')
    verify_domain_plan(value)
    return value


def authorize(stage):
    require(stage != 'final', 'Final fitting/release is not implemented by this frozen backtest driver')
    require(stage in STAGES, 'Only fixed backtest stages allowed')
    value = plan(); c6.verify_hashes(value['stage_dependency_sha256'][stage])
    if stage != 'development':
        frozen = locked(OUT / 'frozen_candidate.json')
        check_later_gate(stage, frozen, sha(OUT / 'candidate_plan.json'))
        require(frozen['source_sha256'] == value['source_sha256']
                and frozen['parameters'] == value['parameters'] and frozen['formulas'] == FORMULAS
                and frozen['thresholds'] == THRESHOLDS
                and frozen['domain_plan_sha256'] == value['domain_plan_sha256'], 'Development freeze changed')
        c6.verify_hashes(frozen['development_artifact_sha256'])
        require(frozen['result_sha256'] == sha(OUT / 'evaluation/development/results.json'),
                'Development result changed')
        development = read_result('development')
        require(development['decision']['passed'] is True
                and development['decision']['selected_candidate'] == CANDIDATE, 'Development did not pass')
    return value


def partition(stage):
    data, X, y, Xq, yq, fi, qi = c6.partition(stage)
    specification = locked(OUT / 'candidate_plan.json')['split_contract'][stage]
    for label, array in [('fit', fi), ('query', qi)]:
        require(specification[label + '_indices_content_sha256'] == value_sha(array, '<i8'),
                'R11 canonical partition differs')
        require(specification[label + '_rows'] == len(array), 'R11 partition rows differ')
    previous = c6.baseline_folder(stage); old = c6.OUT / 'models' / stage / 'geography_catboost'
    e6.verify_indices(previous, fi, qi, 'check_indices.npy')
    e6.verify_indices(old, fi, qi, 'query_indices.npy')
    return data, X, y, Xq, yq, fi, qi


def fit(stage):
    specification = authorize(stage); destination = OUT / 'models' / stage
    require(not destination.exists(), 'Model exists; do not retrain')
    data, X, y, Xq, yq, fi, qi = partition(stage)
    del yq  # No query labels enter either the preparer or model helper.
    geo, geoq, _ = load_geographic_partition(stage, fi, qi)
    weights, contract, _, weight_inputs = load_weights(stage, X, Xq, fi, qi, specification)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + stage + '-' + uuid.uuid4().hex); staging.mkdir()
    started = time.monotonic(); cats = data['categorical_columns']
    print(json.dumps({'event': 'fit_started', 'stage': stage, 'fit_rows': len(X),
                      'query_rows': len(Xq), 'features': 95, 'parameters': specification['parameters'],
                      'sample_weight': 'importance_clip_then_mean_one'}), flush=True)
    model, state, metadata = component.fit_model(X, y, geo, cats, weights, contract)
    predictions = component.predict_model(model, state, Xq, geoq, cats)
    model.save_model(str(staging / 'model.cbm')); write(staging / 'feature_state.json', state)
    reloaded = load_model(staging / 'model.cbm')
    replay = component.predict_model(reloaded, read(staging / 'feature_state.json'),
                                     Xq.iloc[:64], geoq.iloc[:64], cats)
    require(np.allclose(replay, predictions[:64], rtol=1e-12, atol=1e-5), 'Saved model reload differs')
    require(int(reloaded.tree_count_) == 2999, 'Unexpected CatBoost iteration count')
    require(predictions.shape == (len(Xq),) and np.isfinite(predictions).all()
            and (predictions >= 0).all() and predictions.dtype == np.dtype('float64'), 'Invalid TWD predictions')
    for name, values in [('fit_indices', fi), ('query_indices', qi), ('predictions', predictions)]:
        np.save(staging / (name + '.npy'), values, allow_pickle=False)
    metadata.update({
        'round': 11, 'stage': stage, 'component': component.COMPONENT, 'created_at_utc': now(),
        'plan_sha256': sha(OUT / 'candidate_plan.json'), 'source_sha256': specification['source_sha256'],
        'outer_fit_rows': len(X), 'query_rows': len(Xq),
        'fit_target_sha256': value_sha(y, '<f8'),
        'fit_transformed_target_values_sha256': value_sha(np.asarray(y, dtype='float64') / 1e6, '<f8'),
        'target_formula': component.FEATURE_RULES['fit_target'],
        'prediction_formula': component.FEATURE_RULES['prediction'],
        'actual_iterations': int(reloaded.tree_count_),
        'fit_X_sha256': frame_sha(X), 'query_X_sha256': frame_sha(Xq),
        'fit_geo_sha256': frame_sha(geo), 'query_geo_sha256': frame_sha(geoq),
        'fit_indices_content_sha256': value_sha(fi, '<i8'),
        'query_indices_content_sha256': value_sha(qi, '<i8'),
        'domain_plan_sha256': specification['domain_plan_sha256'],
        'preparation_manifest_sha256': contract['preparation_manifest_sha256'],
        'weight_preparation_input_sha256': weight_inputs,
        'geo_manifest_sha256': sha(c6.OUT / 'geography' / stage / 'manifest.json'),
        'artifact_sha256': {p.name: sha(p) for p in staging.iterdir() if p.is_file()},
        'reload_64_rows_verified': True, 'query_labels_used': False, 'prediction_units': 'TWD',
        'direct_component_score_omitted': True,
        'runner_elapsed_seconds': time.monotonic() - started,
        'frozen_candidate_sha256': None if stage == 'development' else sha(OUT / 'frozen_candidate.json'),
    })
    validate_model_contract(metadata, specification, {})
    locked_write(staging / 'metadata.json', metadata)
    authorize(stage)
    verified, current_inputs = verify_weight_manifest(stage, specification)
    require(current_inputs == weight_inputs, 'Preparation inputs changed during fitting')
    require(verified['fit_weight_values_sha256'] == component.weights_values_sha256(weights),
            'Training weights changed')
    publish(staging, destination)
    print(json.dumps({'event': 'fit_completed', 'stage': stage,
                      'elapsed_seconds': time.monotonic() - started}), flush=True)


def model_check(stage, X, y, Xq, fi, qi, geo, geoq, weights, contract, cats, specification, weight_inputs):
    folder = OUT / 'models' / stage; meta = locked(folder / 'metadata.json')
    expected = {key: frame_sha(frame) for key, frame in [
        ('fit_X_sha256', X), ('query_X_sha256', Xq), ('fit_geo_sha256', geo), ('query_geo_sha256', geoq)]}
    expected.update({
        'domain_plan_sha256': specification['domain_plan_sha256'],
        'preparation_manifest_sha256': contract['preparation_manifest_sha256'],
        'weight_contract': contract, 'fit_sample_weight_values_sha256': component.weights_values_sha256(weights),
        'fit_indices_content_sha256': value_sha(fi, '<i8'),
        'query_indices_content_sha256': value_sha(qi, '<i8'),
        'weight_preparation_input_sha256': weight_inputs,
    })
    validate_model_contract(meta, specification, expected)
    require(meta['stage'] == stage and meta['plan_sha256'] == sha(OUT / 'candidate_plan.json'), 'Wrong model stage/plan')
    require(meta['source_sha256'] == specification['source_sha256'], 'Model source closure differs')
    require(meta['fit_transformed_target_values_sha256'] == value_sha(np.asarray(y, dtype='float64') / 1e6, '<f8'),
            'Transformed fit target differs')
    require(meta['outer_fit_rows'] == len(X) and meta['query_rows'] == len(Xq), 'Model sample counts differ')
    require(meta['fit_target_sha256'] == meta['fit_target_values_sha256'] == value_sha(y, '<f8'), 'Fit target identity differs')
    require(meta['geo_manifest_sha256'] == sha(c6.OUT / 'geography' / stage / 'manifest.json'), 'Map version differs')
    require(meta['reload_64_rows_verified'] is True and meta['prediction_units'] == 'TWD', 'Prediction contract differs')
    require(meta['frozen_candidate_sha256'] == (None if stage == 'development' else sha(OUT / 'frozen_candidate.json')),
            'Model candidate-selection identity differs')
    required = {'model.cbm', 'feature_state.json', 'predictions.npy', 'fit_indices.npy', 'query_indices.npy'}
    require(set(meta['artifact_sha256']) == required, 'Saved model artifact inventory differs')
    for name, digest in meta['artifact_sha256'].items():
        require(Path(name).name == name and sha(folder / name) == digest, 'Saved model artifact changed')
    e6.verify_indices(folder, fi, qi, 'query_indices.npy')
    saved = e6.vector(folder / 'predictions.npy', len(qi))
    model = load_model(folder / 'model.cbm')
    require(int(model.tree_count_) == 2999, 'Reloaded model iteration count differs')
    state = read(folder / 'feature_state.json')
    require(component.fit_feature_state(X, geo, cats, weights, contract) == state,
            'Saved geographic/weight state differs from declared fit inputs')
    require(meta['feature_state_content_sha256'] == state['state_content_sha256']
            and meta['geo_state_content_sha256'] == state['geo_state']['state_content_sha256']
            and meta['weight_contract_sha256'] == state['weight_contract_sha256']
            and meta['fit_sample_weight_summary'] == state['fit_sample_weight_summary'], 'State/metadata weights differ')
    replay = component.predict_model(model, state, Xq, geoq, cats)
    require(np.allclose(saved, replay, rtol=1e-12, atol=1e-5), 'All-query model replay differs')
    return saved, {rel(p): sha(p) for p in folder.iterdir() if p.is_file()}


def read_result(stage):
    value = locked(OUT / 'evaluation' / stage / 'results.json')
    require(value['round'] == 11 and value['stage'] == stage
            and value['plan_sha256'] == sha(OUT / 'candidate_plan.json'), 'Wrong saved R11 result')
    require(value['source_sha256'] == sources() and value['formulas'] == FORMULAS
            and value['thresholds'] == THRESHOLDS, 'Saved result protocol differs')
    require(value['domain_plan_sha256'] == sha(OUT / 'domain_plan.json'), 'Saved domain identity differs')
    require(value['frozen_candidate_sha256'] == (None if stage == 'development' else sha(OUT / 'frozen_candidate.json')),
            'Saved result freeze differs')
    require(value['evaluated_candidates'] == [CONTROL, CANDIDATE]
            and value['candidate_component_separately_scored'] is False, 'Candidate inventory differs')
    require(value['query_X_used_for_training_weights'] is True and value['query_labels_used_for_training'] is False,
            'Saved data-use declarations differ')
    require(value['decision'] == decide(stage, value['metrics']), 'Saved gate decision differs')
    for field in ['input_artifact_sha256', 'output_artifact_sha256']:
        c6.verify_hashes(value[field])
    return value


def freeze_development(result):
    path = OUT / 'frozen_candidate.json'; specification = plan()
    artifacts = {rel(p): sha(p) for folder in [OUT / 'models/development', OUT / 'evaluation/development']
                 for p in folder.iterdir() if p.is_file()}
    value = {
        'created_at_utc': now(), 'round': 11,
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'result_sha256': sha(OUT / 'evaluation/development/results.json'),
        'passed': result['decision']['passed'], 'selected_candidate': result['decision']['selected_candidate'],
        'development_artifact_sha256': artifacts, 'source_sha256': sources(),
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'parameters': component.parameters(),
        'domain_plan_sha256': specification['domain_plan_sha256'],
        'later_stages_allowed': result['decision']['passed'], 'historically_untouched': False,
    }
    if path.exists():
        saved = locked(path)
        require(all(saved[k] == v for k, v in value.items() if k != 'created_at_utc'), 'Existing freeze differs')
    else:
        locked_write(path, value)


def evaluate(stage):
    specification = authorize(stage); destination = OUT / 'evaluation' / stage
    if destination.exists():
        result = read_result(stage)
        if stage == 'development':
            freeze_development(result)
        print(json.dumps({'status': 'existing_result_verified', 'stage': stage, 'decision': result['decision']}), flush=True)
        return result
    data, X, y, Xq, yq, fi, qi = partition(stage)
    evaluation_weights = target_weights(data)
    require(evaluation_weights == specification['target_group_weights'], 'Fixed test-X scoring group weights differ')
    geo, geoq, _ = load_geographic_partition(stage, fi, qi)
    training_weights, contract, _, weight_inputs = load_weights(stage, X, Xq, fi, qi, specification)
    new, inputs = model_check(stage, X, y, Xq, fi, qi, geo, geoq, training_weights, contract,
                              data['categorical_columns'], specification, weight_inputs)
    inputs.update(weight_inputs); inputs.update(specification['stage_dependency_sha256'][stage])
    reference, _ = c6.load_reference(stage, X); previous_folder = c6.baseline_folder(stage)
    old = e6.vector(previous_folder / 'catboost_raw.npy', len(qi))
    unit = e6.vector(previous_folder / 'lightgbm_l1_unit63.npy', len(qi))
    geographic = e6.vector(c6.OUT / 'models' / stage / 'geography_catboost/predictions.npy', len(qi))
    original = e6.vector(c6.OUT / 'evaluation' / stage / 'geography_blend.npy', len(qi))
    replay = blend(old, geographic, unit, reference, Xq)
    require(np.allclose(original, replay, rtol=1e-12, atol=1e-5), 'All-row v3 control does not replay')
    predictions = {CONTROL: original, CANDIDATE: blend(old, new, unit, reference, Xq)}
    metrics = {name: score(yq, pred, X, Xq, c6.SPLITS[stage][0], evaluation_weights)
               for name, pred in predictions.items()}
    previous = locked(c6.OUT / 'evaluation' / stage / 'results.json')
    require(metrics[CONTROL] == previous['metrics']['geography_blend'], 'All recomputed v3 metrics differ from archive')
    decision = decide(stage, metrics)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ('.' + stage + '-' + uuid.uuid4().hex); staging.mkdir()
    for name, values in predictions.items():
        np.save(staging / (name + '.npy'), values, allow_pickle=False)
    outputs = {rel(destination / p.name): sha(p) for p in staging.iterdir() if p.is_file()}
    result = {
        'round': 11, 'stage': stage, 'created_at_utc': now(),
        'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'frozen_candidate_sha256': None if stage == 'development' else sha(OUT / 'frozen_candidate.json'),
        'domain_plan_sha256': specification['domain_plan_sha256'],
        'preparation_manifest_sha256': contract['preparation_manifest_sha256'],
        'weight_contract': contract, 'source_sha256': specification['source_sha256'],
        'input_artifact_sha256': inputs, 'output_artifact_sha256': outputs,
        'formulas': FORMULAS, 'thresholds': THRESHOLDS, 'target_group_weights': evaluation_weights,
        'metrics': metrics, 'decision': decision, 'v3_control_replayed_all_rows': True,
        'new_model_replayed_all_rows': True, 'fit_rows': len(fi), 'query_rows': len(qi),
        'evaluated_candidates': [CONTROL, CANDIDATE], 'candidate_component_separately_scored': False,
        'query_X_used_for_training_weights': True, 'query_labels_used_for_training': False,
        'training_importance_weights_used_for_scoring': False,
        'public_score_used': False, 'historically_untouched': False,
        'new_public_score_known': False, 'private_score_known': False,
    }
    locked_write(staging / 'results.json', result)
    authorize(stage)
    _, current_inputs = verify_weight_manifest(stage, specification)
    require(current_inputs == weight_inputs, 'Prepared source changed during evaluation')
    publish(staging, destination)
    if stage == 'development':
        freeze_development(result)
    print(json.dumps({'stage': stage, 'metrics': {k: {n: v[n] for n in ['overall_mae', 'weighted_mae']}
                                                for k, v in metrics.items()}, 'decision': decision}), flush=True)
    return result


def final_outcome(results, failed_weight_stages):
    """A method stop needs verified negative evidence, not a missing price file."""
    require('development' in results and set(results) <= set(STAGES), 'Invalid finalized stage inventory')
    require(set(failed_weight_stages) <= {'confirmation', 'auxiliary'}, 'Invalid pure-X stopping stage')
    require(not set(failed_weight_stages) & set(results), 'A failed-weight stage cannot have a price score')
    if failed_weight_stages:
        require(results['development']['decision']['passed'] is True, 'Later preparation was not authorized')
        return False, 'pure_X_feasibility_failed'
    if not results['development']['decision']['passed']:
        require(set(results) == {'development'}, 'Development rejection has later scores')
        return False, 'development_price_gates_failed'
    require(set(results) == set(STAGES), 'Missing later price results are not a method rejection')
    accepted = all(value['decision']['passed'] for value in results.values())
    return accepted, 'three_stage_price_gates_passed' if accepted else 'later_price_gates_failed'


def require_no_staging(entries=None):
    """Incomplete executions cannot be reported as an unexecuted stage."""
    if entries is None:
        entries = [(area, path.name) for area in ('weights', 'models', 'evaluation')
                   if (OUT / area).exists() for path in (OUT / area).iterdir()]
    unexpected = [area + '/' + name for area, name in entries
                  if any(name.startswith(('_' + stage + '_') if area == 'weights'
                                         else ('.' + stage + '-')) for stage in STAGES)]
    require(not unexpected, 'Incomplete staged execution requires inspection: ' + ', '.join(unexpected))


def finalize():
    require_no_staging()
    specification = plan(); frozen = locked(OUT / 'frozen_candidate.json')
    development = read_result('development')
    require(frozen['plan_sha256'] == sha(OUT / 'candidate_plan.json')
            and frozen['source_sha256'] == specification['source_sha256']
            and frozen['domain_plan_sha256'] == specification['domain_plan_sha256'], 'Finalization freeze differs')
    require(frozen['result_sha256'] == sha(OUT / 'evaluation/development/results.json')
            and frozen['passed'] == development['decision']['passed']
            and frozen['selected_candidate'] == development['decision']['selected_candidate'], 'Development freeze differs')
    c6.verify_hashes(frozen['development_artifact_sha256'])
    results = {'development': development}
    statuses = {'development': 'price_evaluated'}
    failed_weight_stages = []
    preparation_evidence = {}; preparation_manifest_hashes = {}
    if frozen['passed']:
        for stage in ('confirmation', 'auxiliary'):
            authorize(stage)
            folder = OUT / 'weights' / stage; path = folder / 'manifest.json'
            model_folder = OUT / 'models' / stage; result_folder = OUT / 'evaluation' / stage
            if path.exists():
                # Only this non-scoring finalizer accepts valid failed feasibility.
                # Missing files, invalid receipts and program errors still raise.
                manifest, evidence = verify_weight_manifest(stage, specification, require_pass=False)
                preparation_evidence.update(evidence)
                preparation_manifest_hashes[stage] = sha(path)
                if manifest['feasibility_passed'] is False:
                    require(not model_folder.exists() and not result_folder.exists(),
                            'Failed pure-X preparation has unauthorized price artifacts')
                    failed_weight_stages.append(stage)
                    statuses[stage] = 'pure_X_feasibility_failed_price_not_evaluated'
                elif result_folder.exists():
                    require(model_folder.exists(), 'Price result is missing its model')
                    results[stage] = read_result(stage)
                    statuses[stage] = 'price_evaluated'
                else:
                    require(not model_folder.exists(), 'Unscored model is an incomplete run, not a method rejection')
                    statuses[stage] = 'weights_prepared_price_not_executed'
            else:
                require(not folder.exists() and not path.with_suffix('.sha256').exists()
                        and not model_folder.exists() and not result_folder.exists(),
                        'Missing preparation manifest with partial artifacts is an error')
                statuses[stage] = 'not_executed'
    else:
        for stage in ('confirmation', 'auxiliary'):
            require(not (OUT / 'weights' / stage).exists() and not (OUT / 'models' / stage).exists()
                    and not (OUT / 'evaluation' / stage).exists(), 'Rejected development has unauthorized later artifacts')
            statuses[stage] = 'not_executed_after_development_rejection'
    accepted, reason = final_outcome(results, failed_weight_stages)
    if failed_weight_stages:
        for stage in ('confirmation', 'auxiliary'):
            if statuses[stage] == 'not_executed':
                statuses[stage] = 'not_executed_after_pure_X_stop'
    c6.verify_hashes(preparation_evidence)
    plan()
    result = {
        'created_at_utc': now(), 'round': 11, 'plan_sha256': sha(OUT / 'candidate_plan.json'),
        'frozen_candidate_sha256': sha(OUT / 'frozen_candidate.json'),
        'domain_plan_sha256': specification['domain_plan_sha256'],
        'accepted': accepted, 'candidate': CANDIDATE, 'selected_candidate': frozen['selected_candidate'],
        'deployment_decision': 'backtest_accepted_pending_final_release' if accepted else 'keep_v3',
        'stopping_reason': reason, 'stage_status': statuses,
        'pure_X_feasibility_failure_stages': failed_weight_stages,
        'stage_preparation_manifest_sha256': preparation_manifest_hashes,
        'preparation_evidence_sha256': preparation_evidence,
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
        require(all(saved[k] == v for k, v in result.items() if k != 'created_at_utc'), 'Existing final decision differs')
        result = saved
    else:
        locked_write(path, result)
    print(json.dumps(result), flush=True)
    return result


def additional_self_checks():
    import copy
    import pandas as pd
    checks = []
    def rejected(name, operation):
        try:
            operation()
        except (RuntimeError, ValueError, AssertionError, KeyError):
            checks.append(name)
        else:
            raise AssertionError('Expected rejection: ' + name)
    require_no_staging([('weights', 'development'), ('models', 'confirmation'), ('evaluation', 'auxiliary')])
    checks.append('published_stage_directories_are_not_partial_runs')
    for area, name in [('weights', '_confirmation_savedfailure'), ('models', '.auxiliary-savedfailure'), ('evaluation', '.development-savedfailure')]:
        rejected('finalization_rejects_partial_' + area,
                 lambda area=area, name=name: require_no_staging([(area, name)]))
    selected = {'passed': True, 'later_stages_allowed': True,
                'selected_candidate': CANDIDATE, 'plan_sha256': 'fixed'}
    check_later_gate('confirmation', selected, 'fixed'); check_later_gate('auxiliary', selected, 'fixed')
    checks.append('later_stages_require_own_passed_frozen_candidate')
    for key, value in [('passed', False), ('later_stages_allowed', False),
                       ('selected_candidate', CONTROL), ('plan_sha256', 'different')]:
        bad = dict(selected); bad[key] = value
        rejected('later_gate_rejects_' + key, lambda bad=bad: check_later_gate('confirmation', bad, 'fixed'))
    rejected('final_not_implemented', lambda: check_later_gate('final', selected, 'fixed'))
    specification = {
        'domain_plan_sha256': '1' * 64, 'domain_runtime_versions': {'python': 'fixed'},
        'domain_config': {'fixed': True},
        'split_contract': {'development': {'fit_indices_content_sha256': 'fit', 'query_indices_content_sha256': 'query'}},
        'source_sha256': {name: 'source' for name in EXTRA_SOURCES}, 'parameters': component.parameters(),
    }
    import round11_domain_weights as domain
    synthetic_summary = {'largest_uniform_group_row_share': .01, 'largest_group_weight_share': .01,
                         'clipped_fraction': 0., 'row_ess_fraction': 1., 'group_ess_ratio_to_uniform': 1.}
    synthetic_coverage = {key: {'adequate_query_fraction': 1.} for key in ['region', 'structure']}
    manifest = {
        'round': 11, 'stage': 'development', 'domain_plan_sha256': '1' * 64,
        'runtime_versions': {'python': 'fixed'}, 'config': {'fixed': True},
        'feasibility_passed': True, 'feasibility_checks': domain.feasibility_checks(synthetic_summary, synthetic_coverage),
        'weight_summary': synthetic_summary, 'coverage': synthetic_coverage,
        'fit_indices_content_sha256': 'fit', 'query_indices_content_sha256': 'query',
        'source_sha256': dict(specification['source_sha256']),
        'query_X_used_for_weights': True, 'query_y_or_price_cache_used': False,
        'original_features_mutated': False,
    }
    validate_prepared_contract(manifest, specification, 'development')
    checks.append('pure_X_manifest_contract')
    late_failed = copy.deepcopy(manifest)
    late_failed['weight_summary']['row_ess_fraction'] = .49
    late_failed['feasibility_checks'] = domain.feasibility_checks(late_failed['weight_summary'], late_failed['coverage'])
    late_failed['feasibility_passed'] = False
    validate_prepared_contract(late_failed, specification, 'development', require_pass=False)
    checks.append('verified_failed_feasibility_only_allowed_for_non_scoring_inspection')
    rejected('default_training_path_rejects_failed_feasibility',
             lambda: validate_prepared_contract(late_failed, specification, 'development'))
    inconsistent = copy.deepcopy(late_failed); inconsistent['feasibility_passed'] = True
    rejected('inconsistent_feasibility_flag_rejected_even_for_finalizer',
             lambda: validate_prepared_contract(inconsistent, specification, 'development', require_pass=False))
    inconsistent = copy.deepcopy(late_failed); inconsistent['feasibility_checks'] = manifest['feasibility_checks']
    rejected('inconsistent_feasibility_checks_rejected_even_for_finalizer',
             lambda: validate_prepared_contract(inconsistent, specification, 'development', require_pass=False))
    passed_result = {'decision': {'passed': True}}
    accepted, reason = final_outcome({'development': passed_result}, ['confirmation'])
    require(not accepted and reason == 'pure_X_feasibility_failed', 'Failed late feasibility was accepted')
    checks.append('late_pure_X_failure_rejects_without_unperformed_price_scores')
    rejected('missing_later_result_is_error_not_method_rejection',
             lambda: final_outcome({'development': passed_result}, []))
    rejected('scored_failed_weight_stage_rejected',
             lambda: final_outcome({'development': passed_result, 'confirmation': passed_result}, ['confirmation']))
    require(final_outcome({stage: passed_result for stage in STAGES}, [])[0] is True,
            'Three actual price passes are required for acceptance')
    checks.append('acceptance_requires_three_actual_price_passes')
    for key, value in [('stage', 'confirmation'), ('domain_plan_sha256', 'wrong'),
                       ('fit_indices_content_sha256', 'wrong'), ('query_indices_content_sha256', 'wrong'),
                       ('feasibility_passed', False), ('feasibility_checks', {'fixed_check': False}),
                       ('source_sha256', {}), ('config', {}), ('query_y_or_price_cache_used', True)]:
        bad = copy.deepcopy(manifest); bad[key] = value
        rejected('preparation_rejects_' + key, lambda bad=bad: validate_prepared_contract(bad, specification, 'development'))
    X = pd.DataFrame({'value': [1.0, 2.0]}, index=[5, 9])
    Xq = pd.DataFrame({'value': [3.0, 4.0]}, index=[13, 17])
    fi = np.array([10, 11], dtype='int64'); qi = np.array([12, 13], dtype='int64')
    weights = np.array([0.5, 1.5], dtype='float64')
    contract = {
        'phase': 'development', 'fit_X_sha256': frame_sha(X), 'query_X_sha256': frame_sha(Xq),
        'fit_indices_content_sha256': value_sha(fi, '<i8'), 'query_indices_content_sha256': value_sha(qi, '<i8'),
        'weights_values_sha256': component.weights_values_sha256(weights), 'domain_plan_sha256': '1' * 64,
        'preparation_manifest_sha256': '2' * 64, 'query_X_used_for_weights': True,
        'query_labels_used_for_weights': False,
    }
    validate_weight_identity(contract, 'development', X, Xq, fi, qi, weights, specification, '2' * 64)
    checks.append('whole_query_fit_indices_and_ordered_weights_bound')
    for key in ['phase', 'query_X_sha256', 'fit_indices_content_sha256', 'query_indices_content_sha256',
                'domain_plan_sha256', 'preparation_manifest_sha256', 'weights_values_sha256']:
        bad = dict(contract); bad[key] = 'wrong'
        rejected('weight_contract_rejects_' + key, lambda bad=bad: validate_weight_identity(
            bad, 'development', X, Xq, fi, qi, weights, specification, '2' * 64))
    rejected('changed_query_batch_rejected', lambda: validate_weight_identity(
        contract, 'development', X, Xq.iloc[::-1], fi, qi, weights, specification, '2' * 64))
    rejected('changed_weights_order_rejected', lambda: validate_weight_identity(
        contract, 'development', X, Xq, fi, qi, weights[::-1], specification, '2' * 64))
    rejected('floating_indices_rejected', lambda: validate_weight_identity(
        contract, 'development', X, Xq, fi.astype(float), qi, weights, specification, '2' * 64))
    meta = {
        'round': 11, 'component': component.COMPONENT, 'parameters': component.parameters(),
        'feature_count': 95, 'categorical_feature_count': 23, 'iterations': 2999, 'actual_iterations': 2999,
        'target_mode': 'raw', 'sample_weight': 'importance_clip_then_mean_one',
        'feature_rules': component.FEATURE_RULES, 'target_formula': component.FEATURE_RULES['fit_target'],
        'prediction_formula': component.FEATURE_RULES['prediction'],
        'query_X_used_for_training_weights': True, 'query_labels_used': False,
        'target_used_for_geographic_state': False, 'query_used_for_geographic_state': False,
        'reference_features_changed': False, 'reference_or_trend_applied_here': False,
        'fit_uses_eval_set': False, 'early_stopping_used': False, 'weight_contract': contract,
    }
    validate_model_contract(meta, specification, {'weight_contract': contract})
    checks.append('95_raw_weighted_model_metadata_contract')
    for key, value in [('feature_count', 109), ('sample_weight', 'uniform'), ('target_mode', 'unit'),
                       ('query_X_used_for_training_weights', False), ('query_labels_used', True),
                       ('query_used_for_geographic_state', True), ('fit_uses_eval_set', True),
                       ('reference_or_trend_applied_here', True)]:
        bad = copy.deepcopy(meta); bad[key] = value
        rejected('model_rejects_' + key, lambda bad=bad: validate_model_contract(bad, specification, {'weight_contract': contract}))
    return checks


def self_check(write_output=True):
    import copy
    base={'overall_mae':100.,'weighted_mae':100.,'strata':{'seen':{'rows':500,'mae':100.},'missing':{'rows':0,'mae':None}},'forecast_quarters':{q:{'rows':10,'mae':100.} for q in ['Q1','Q2','Q3','Q4']}}
    checks=[]
    component_checks=component.self_check()
    def run(label,truth):require(truth,label);checks.append(label)
    def metrics(overall,weighted=None):
        value=copy.deepcopy(base);value['overall_mae']=overall;value['weighted_mae']=overall if weighted is None else weighted
        return {CONTROL:copy.deepcopy(base),CANDIDATE:value}
    run('development_exact_1pct_passes',decide('development',metrics(99))['passed'])
    run('development_below_1pct_rejected',not decide('development',metrics(99.00001))['passed'])
    run('weighted_also_required',not decide('development',metrics(98,100))['passed'])
    m=metrics(98);m[CANDIDATE]['strata']['seen']['mae']=105.00001
    run('500_row_stratum_gate',not decide('development',m)['passed'])
    m=metrics(98);m[CANDIDATE]['forecast_quarters']['Q1']['mae']=105.00001
    run('quarter_gate',not decide('development',m)['passed'])
    m=metrics(99);m[CANDIDATE]['forecast_quarters']['Q1']['mae']=99
    run('confirmation_one_quarter_insufficient',not decide('confirmation',m)['passed'])
    m[CANDIDATE]['forecast_quarters']['Q2']['mae']=99
    run('confirmation_two_quarters_pass',decide('confirmation',m)['passed'])
    run('auxiliary_exact_3pct_passes',decide('auxiliary',metrics(103))['passed'])
    run('auxiliary_over_3pct_rejected',not decide('auxiliary',metrics(103.00001))['passed'])
    calls=[];query=[object(),object()]
    def fake(reference,X,values,variant):
        require(X is query and reference=='fit_only' and variant=='trend','Reference call changed');calls.append(values.copy());return values*1.1
    result=blend([1,2],[3,4],[5,6],'fit_only',query,fake)
    run('fixed_weights_and_single_trend',len(calls)==1 and np.allclose(result,1.1*(.375*np.array([1,2])+.375*np.array([3,4])+.25*np.array([5,6]))))
    extras=additional_self_checks()
    checks.extend(extras)
    proof={'created_at_utc':now(),'all_checks_passed':True,'synthetic_checks':checks,'component_synthetic_checks':component_checks,'source_sha256':sources(),'real_model_training_or_scoring':False,'parameters':component.parameters()}
    if write_output:
        path=OUT/'audit/preflight.json';require(not path.exists(),'Preflight exists; preserve audit')
        write(path,proof)
    print(json.dumps(proof),flush=True)
    return proof



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['self-check', 'plan', 'fit', 'evaluate', 'finalize'])
    parser.add_argument('--stage', choices=(*STAGES, 'final'), default='development')
    parser.add_argument('--no-write', action='store_true', help='Synthetic checks only; do not publish preflight')
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
