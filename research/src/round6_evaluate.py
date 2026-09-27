"""Immutable round-six evaluation, single-candidate freeze and acceptance gates.

No evaluation or training runs on import. Predictions remain unrounded TWD.
The historical round-three confirmation baseline has already been evaluated;
this protocol does not represent any historical period as untouched labels.
"""
from __future__ import annotations
import argparse
import ast
import csv
import hashlib
import io
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import round6_common as common
from round6_geo import FEATURE_RULES, fit_feature_state, augment_features, frame_sha
from round6_prepare_geo import load_geographic_partition
from model_common import temporal_indices
from round2_neighborhood import predict_reference
from round3_evaluate import score, target_weights, runtime_versions, GROUPS, QUARTERS

ROOT, OUT = common.ROOT, common.OUT
CONTROL = 'v2_control'
FREEZE = OUT / 'frozen_candidate.json'
ACCEPTANCE = OUT / 'accepted_candidate.json'
FORMULAS = {
    CONTROL: {'base': '0.75 * old_catboost + 0.25 * lightgbm_unit', 'reference': 'v2_trend_once'},
    'geography_blend': {'base': '0.75 * (0.5 * old_catboost + 0.5 * geography_catboost) + 0.25 * lightgbm_unit', 'reference': 'v2_trend_once'},
}
REQUIRED_COMPONENTS = {CONTROL: [], 'geography_blend': ['geography_catboost']}
THRESHOLDS = {
    'development_min_overall_improvement': .01, 'development_min_weighted_improvement': .01,
    'development_max_quarter_deterioration': .05, 'large_stratum_min_rows': 500,
    'max_large_stratum_deterioration': .05, 'confirmation_max_overall_deterioration': .0,
    'confirmation_max_weighted_deterioration': .0, 'confirmation_min_improved_quarters': 2,
    'auxiliary_max_overall_deterioration': .03, 'auxiliary_max_weighted_deterioration': .03,
    'auxiliary_quarters_diagnostic_only': True,
}
require, read, rel, sha256 = common.require, common.read, common.rel, common.sha256


def now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes():
    """Freeze the local import closure, including fitting helpers and scoring code."""
    pending = [ROOT / 'src' / name for name in (*common.FIT_SOURCES, Path(__file__).name)]
    found = set()
    while pending:
        path = pending.pop().resolve()
        if path in found:
            continue
        require(path.is_file(), 'Missing fit/evaluation source: ' + str(path))
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names = [item.name for item in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                local = ROOT / 'src' / (name.split('.')[0] + '.py')
                if local.is_file() and local.resolve() not in found:
                    pending.append(local)
    return {rel(path): sha256(path) for path in sorted(found)}


def checked_plan():
    value = common.plan()
    require(value['candidate_formulas'] == FORMULAS, 'Plan candidate formulas differ')
    require(value['required_components'] == REQUIRED_COMPONENTS, 'Plan required components differ')
    require(value['thresholds'] == THRESHOLDS, 'Plan evaluation thresholds differ')
    require(tuple(common.COMPONENTS) == ('geography_catboost',), 'Only the one predeclared component is allowed')
    require(value['components']['geography_catboost']['feature_rules'] == FEATURE_RULES, 'Plan geographic feature rules differ')
    evaluation_sources = value['evaluation_source_sha256']
    require(evaluation_sources == source_hashes(), 'Planned evaluation source closure differs')
    require(evaluation_sources.get(rel(Path(__file__))) == sha256(Path(__file__)), 'Evaluator was not frozen in plan')
    common.verify_hashes(evaluation_sources)
    return value


def data_hashes(plan):
    result = dict(plan['data_sha256'])
    paths = [ROOT / 'data_cache' / (split + suffix) for split in ['train','valid','test']
             for suffix in ['.pkl','_features.pkl','.json']]
    paths += [ROOT / 'reports/feature_schema.json', ROOT / 'runs/v1/models/final_config.json']
    paths += list((ROOT / 'data_cache').glob('*features*.json'))
    paths += [path for path in [ROOT / 'requirements.txt', ROOT / 'requirements-round2.txt'] if path.exists()]
    for path in paths:
        key, digest = rel(path), sha256(path)
        require(key not in result or result[key] == digest, 'Plan data changed: ' + key)
        result[key] = digest
    return result


def split_contract(data):
    result = {}
    for stage, (start, end) in common.SPLITS.items():
        fi, qi = temporal_indices(data, start, end=end)
        def digest(array):
            buffer = io.BytesIO(); np.save(buffer, np.asarray(array, dtype='<i8'), allow_pickle=False)
            return hashlib.sha256(buffer.getvalue()).hexdigest()
        require(not np.intersect1d(fi, qi).size, 'Fit/query index overlap')
        result[stage] = {'dates': [start,end], 'fit_rows': len(fi), 'query_rows': len(qi),
                         'fit_indices_npy_sha256': digest(fi), 'query_indices_npy_sha256': digest(qi)}
    return result


def files_hashes(folder):
    return {rel(path): sha256(path) for path in sorted(Path(folder).rglob('*'))
            if path.is_file() and path.suffix in {'.npy','.json','.sha256','.pkl','.cbm','.txt','.csv'}
            and 'catboost_info' not in path.parts}


def vector(path, count):
    result = np.asarray(np.load(path, allow_pickle=False), dtype='float64')
    require(result.shape == (count,) and np.isfinite(result).all() and (result >= 0).all(), 'Invalid predictions: ' + str(path))
    return result


def verify_indices(folder, fi, qi, query_filename):
    for filename, expected in [('fit_indices.npy',fi),(query_filename,qi)]:
        actual = np.load(folder / filename, allow_pickle=False)
        require(actual.dtype.kind in 'iu' and np.array_equal(actual, expected), 'Index contents/order differ: ' + str(folder / filename))


def verify_baseline(stage, data, X, Xq, fi, qi, frozen):
    folder = common.baseline_folder(stage)
    verify_indices(folder, fi, qi, 'check_indices.npy')
    hashes = files_hashes(folder)
    if stage in ('development','confirmation'):
        path = ROOT / 'runs/round3_experiments/snapshot_manifest.json'
        manifest = common.read_locked(path)
        for name, digest in list(hashes.items()):
            record = manifest['copied_files'].get(name)
            require(record is not None and record['original_sha256'] == digest and record['backup_sha256'] == digest,
                    'Historical baseline does not match round-three snapshot: ' + name)
            backup = (ROOT / record['backup_path']).resolve()
            require(backup.is_relative_to((ROOT / 'runs/round3_experiments').resolve()) and sha256(backup) == digest, 'Historical baseline backup differs')
        hashes.update({rel(path): sha256(path), rel(path.with_suffix('.sha256')): sha256(path.with_suffix('.sha256'))})
    for name in ['catboost_raw','lightgbm_l1_unit63','v2_control']:
        meta = read(folder / (name + '.json'))
        require(meta['source_sha256'] == data['metadata']['source_sha256'], 'Baseline input source differs')
        require(meta['fit_uses_eval_set'] is False, 'Baseline query-label tuning prohibited')
        require(meta['model_sha256'] == sha256(ROOT / meta['model_path']), 'Baseline model SHA differs')
        require(meta['prediction_sha256'] == sha256(folder / (name + '.npy')), 'Baseline prediction SHA differs')
        if stage == 'auxiliary':
            verify_common_metadata(meta, stage, name, data, fi, qi, folder, 'check_indices.npy', frozen)
        else:
            require(meta['split'] == stage and [meta['start'],meta['end_inclusive']] == list(common.SPLITS[stage]), 'Historical baseline dates differ')
            require(meta['fit_rows'] == len(X) and meta['check_rows'] == len(Xq), 'Historical baseline rows differ')
            require(meta['iterations_selected_from_confirmation'] is False, 'Historical baseline used query stopping')
            require(meta['script_sha256'] == sha256(ROOT / 'src/round3_backtest.py'), 'Historical fit script differs')
            require(meta['feature_source_sha256'] == sha256(ROOT / 'src/house_features.py'), 'Historical features differ')
    reference, path = common.load_reference(stage, X)
    require(rel(path) in hashes, 'Reference artifact missing from provenance')
    return reference, hashes


def verify_common_metadata(meta, stage, component, data, fi, qi, folder, query_file, frozen):
    require(meta['stage'] == stage and meta['component'] == component, 'Wrong component/stage metadata')
    require(meta['dates'] == list(common.SPLITS[stage]), 'Component date window differs')
    require(meta['outer_fit_rows'] == len(fi) and meta['query_rows'] == len(qi), 'Component row count differs')
    require(meta['plan_sha256'] == sha256(OUT / 'candidate_plan.json'), 'Component plan differs')
    require(meta['source_sha256'] == data['metadata']['source_sha256'], 'Component input source differs')
    require(meta['fit_source_sha256'] == common.fit_source_hashes(), 'Component fitting source differs')
    for flag in ['fit_uses_eval_set','early_stopping_used','query_labels_used']:
        require(meta[flag] is False, 'Query-side fitting/tuning prohibited: ' + flag)
    require(meta['fit_indices_sha256'] == sha256(folder / 'fit_indices.npy'), 'Fit-index SHA differs')
    require(meta['query_indices_sha256'] == sha256(folder / query_file), 'Query-index SHA differs')
    if stage != 'development':
        require(frozen is not None and meta['frozen_candidate_sha256'] == sha256(FREEZE), 'Component predates/wrong frozen candidate')
        if 'created_at_utc' in meta:
            require(datetime.fromisoformat(meta['created_at_utc']) >= datetime.fromisoformat(frozen['freeze_timestamp_utc']), 'Component creation precedes freeze')
    else:
        require(meta['frozen_candidate_sha256'] is None, 'Development component has unexpected freeze')


def verify_geo_state(X,Xq,geo,geoq,state):
    require(state['feature_rules']==FEATURE_RULES,'Saved geography rules differ')
    fit_before,query_before=X.copy(deep=True),Xq.copy(deep=True)
    geo_before,geoq_before=geo.copy(deep=True),geoq.copy(deep=True)
    before=json.dumps(state,ensure_ascii=False,sort_keys=True,allow_nan=False)
    rebuilt=fit_feature_state(X,geo)
    require(rebuilt==state,'Saved geographic state differs from exact fit X and geographic features')
    require(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False,'Geographic state used target/query')
    augmented=augment_features(Xq,geoq)
    require(augmented[Xq.columns].equals(Xq) and len(augmented.columns)==95,'Query feature append changed original X')
    require(X.equals(fit_before) and Xq.equals(query_before) and geo.equals(geo_before) and geoq.equals(geoq_before),'Geographic augmentation changed inputs')
    require(json.dumps(state,ensure_ascii=False,sort_keys=True,allow_nan=False)==before,'Query changed saved feature state')
    return {'fit_X_state_reconstruction_verified':True,'query_geographic_state_unchanged':True,'reference_uses_original_features':True}


def verify_component(stage,component,data,X,Xq,fi,qi,reference_path,frozen):
    require(component=='geography_catboost','Unplanned round-six component')
    folder=OUT/'models'/stage/component;meta=read(folder/'metadata.json')
    verify_indices(folder,fi,qi,'query_indices.npy')
    verify_common_metadata(meta,stage,component,data,fi,qi,folder,'query_indices.npy',frozen)
    require(meta['model_sha256']==sha256(folder/'model.cbm'),'New model SHA differs')
    require(meta['prediction_sha256']==sha256(folder/'predictions.npy'),'New prediction SHA differs')
    require(meta['script_sha256']==sha256(ROOT/'src/round6_geo.py'),'Model geographic source differs')
    require(meta['reference_path']==rel(reference_path) and meta['reference_sha256']==sha256(reference_path),'Component reference differs')
    require(meta['prediction_units']=='TWD' and meta['direct_component_mae_omitted'] is True,'Unexpected prediction contract')
    require(meta['reload_64_rows_verified'] is True,'Model reload check missing')
    require(meta['feature_count']==95 and meta['categorical_feature_count']==23,'Geographic component schema differs')
    require(meta['iterations']==2999 and meta['target_used_for_geographic_state'] is False and meta['query_used_for_geographic_state'] is False,'Geographic fitting contract differs')
    require(meta['reference_features_changed'] is False and meta['geo_feature_rules']==FEATURE_RULES,'Reference/geographic rules changed')
    params=meta['parameters']
    require(params==read(OUT/'candidate_plan.json')['components'][component]['parameters'],'Parameters differ from predeclared plan')
    if frozen is not None:require(params==frozen['component_parameters'][component],'Parameters changed after selection')
    state_path=folder/'feature_state.json';state=read(state_path)
    require(meta['feature_state_sha256']==sha256(state_path) and meta['feature_state_content_sha256']==state['state_content_sha256'],'Feature state identity differs')
    target=np.asarray(data['y_train'],dtype='<f8')[fi]
    require(meta['fit_target_values_sha256']==hashlib.sha256(target.tobytes()).hexdigest(),'Fit target identity differs')
    geo,geoq,manifest=load_geographic_partition(stage,fi,qi)
    manifest_path=OUT/'geography'/stage/'manifest.json'
    require(meta['geography_manifest_path']==rel(manifest_path) and meta['geography_manifest_sha256']==sha256(manifest_path),'Geographic dataset manifest differs')
    require(meta['geography_snapshot_year']==manifest['snapshot_year'] and meta['geography_replication_timestamp']==manifest['replication_timestamp'],'Geographic version changed')
    require(meta['fit_geo_feature_values_sha256']==frame_sha(geo) and meta['query_geo_feature_values_sha256']==frame_sha(geoq),'Geographic row values differ')
    verify_geo_state(X,Xq,geo,geoq,state)
    provenance={**files_hashes(folder),**files_hashes(manifest_path.parent),**manifest['source_sha256']}
    return vector(folder/'predictions.npy',len(qi)),params,provenance


def build_predictions(names, old, unit, components, reference, query):
    """Apply v2 trend once to each fixed TWD base; reference always receives original X."""
    require(len(names) == len(set(names)) and set(names).issubset(FORMULAS), 'Unplanned or duplicated candidate')
    result = {}
    for name in names:
        if name == CONTROL:
            base = .75 * old + .25 * unit
        elif name == 'geography_blend':
            base = .75 * (.5 * old + .5 * components['geography_catboost']) + .25 * unit
        else:
            raise RuntimeError('Unknown candidate')
        prediction = np.asarray(predict_reference(reference, query, base, variant='trend'), dtype='float64')
        require(prediction.shape == (len(query),) and np.isfinite(prediction).all() and (prediction >= 0).all(), 'Invalid composite prediction')
        result[name] = prediction
    return result

def large_strata_ok(value, base):
    return all(value['strata'][group]['mae'] <= (1 + THRESHOLDS['max_large_stratum_deterioration']) * item['mae']
               for group, item in base['strata'].items() if item['rows'] >= THRESHOLDS['large_stratum_min_rows'])


def development_selection(metrics):
    require(set(metrics) == set(FORMULAS), 'Exactly the one fixed candidate and control must be evaluated')
    base, checks = metrics[CONTROL], {}
    for name in sorted(metrics):
        if name == CONTROL:
            continue
        value = metrics[name]
        rules = {'overall_improves_at_least_1pct': value['overall_mae'] <= .99 * base['overall_mae'],
                 'weighted_improves_at_least_1pct': value['weighted_mae'] <= .99 * base['weighted_mae'],
                 'quarters_deteriorate_at_most_5pct': all(value['forecast_quarters'][q]['mae'] <= 1.05 * item['mae'] for q,item in base['forecast_quarters'].items() if item['rows']),
                 'large_strata_deteriorate_at_most_5pct': large_strata_ok(value, base)}
        checks[name] = {'eligible': all(rules.values()), 'checks': rules}
    eligible = [name for name, record in checks.items() if record['eligible']]
    selected = min(eligible, key=lambda name: (metrics[name]['weighted_mae'], name)) if eligible else CONTROL
    return {'selected_candidate': selected, 'qualified_candidates': eligible, 'candidate_checks': checks,
            'selection_rule': 'lowest weighted MAE among eligible candidates, name tie-break; otherwise v2_control'}


def stage_decision(stage, metrics, selected):
    require(stage in ('confirmation','auxiliary'), 'Invalid confirmation stage')
    require(selected in FORMULAS and set(metrics) == {CONTROL,selected}, 'Only the frozen candidate and control may be scored')
    base, value = metrics[CONTROL], metrics[selected]
    prefix = stage + '_max_'
    rules = {'overall_within_limit': value['overall_mae'] <= (1 + THRESHOLDS[prefix + 'overall_deterioration']) * base['overall_mae'],
             'weighted_within_limit': value['weighted_mae'] <= (1 + THRESHOLDS[prefix + 'weighted_deterioration']) * base['weighted_mae'],
             'large_strata_deteriorate_at_most_5pct': large_strata_ok(value, base)}
    improved = [q for q,item in base['forecast_quarters'].items() if item['rows'] and value['forecast_quarters'][q]['mae'] < item['mae']]
    if stage == 'confirmation':
        rules['at_least_two_quarters_improve'] = len(improved) >= THRESHOLDS['confirmation_min_improved_quarters']
    return {'selected_candidate': selected, 'passed': selected != CONTROL and all(rules.values()), 'checks': rules,
            'improved_quarters': improved, 'candidate_reselection_performed': False}


def checked_frozen(plan):
    frozen = common.read_locked(FREEZE)
    require(frozen['plan_sha256'] == sha256(OUT / 'candidate_plan.json'), 'Frozen plan differs')
    require(frozen['candidate_formulas'] == FORMULAS and frozen['thresholds'] == THRESHOLDS, 'Frozen formulas/rules differ')
    require(frozen['geo_feature_rules'] == FEATURE_RULES, 'Frozen geographic feature rules differ')
    require(frozen['required_components'] == REQUIRED_COMPONENTS[frozen['selected_candidate']], 'Frozen component list differs')
    require(frozen['source_sha256'] == source_hashes(), 'Frozen source set differs')
    require(frozen['data_sha256'] == data_hashes(plan), 'Frozen input set differs')
    require(frozen['runtime_versions'] == runtime_versions(), 'Frozen runtime differs')
    common.verify_hashes(frozen['source_sha256']); common.verify_hashes(frozen['data_sha256'])
    common.verify_hashes(frozen['development_artifact_sha256'])
    return frozen


def checked_result(stage, plan, frozen=None):
    path = OUT / 'evaluation' / stage / 'results.json'
    report = common.read_locked(path)
    require(report['stage'] == stage and report['plan_sha256'] == sha256(OUT / 'candidate_plan.json'), 'Saved result stage/plan differs')
    require(report['candidate_formulas'] == FORMULAS and report['thresholds'] == THRESHOLDS, 'Saved rules differ')
    require(report['geo_feature_rules'] == FEATURE_RULES, 'Saved geographic feature rules differ')
    require(report['source_sha256'] == source_hashes() and report['data_sha256'] == data_hashes(plan), 'Saved source/input set differs')
    require(report['runtime_versions'] == runtime_versions(), 'Saved runtime differs')
    for key in ['source_sha256','data_sha256','input_artifact_sha256','output_artifact_sha256']:
        common.verify_hashes(report[key])
    if stage == 'development':
        require(report['evaluated_candidates'] == list(FORMULAS), 'Development candidate list differs')
        expected = development_selection(report['metrics'])
    else:
        require(frozen is not None and report['frozen_candidate_sha256'] == sha256(FREEZE), 'Result belongs to another freeze')
        require(report['evaluated_candidates'] == [CONTROL,frozen['selected_candidate']], 'Unauthorized candidate was evaluated')
        require(report['target_group_weights'] == frozen['target_group_weights'] and report['splits'] == frozen['splits'], 'Result split/group contract changed')
        expected = stage_decision(stage, report['metrics'], frozen['selected_candidate'])
    require(report['decision'] == expected, 'Saved acceptance/selection decision differs')
    return report


def evaluate(stage):
    plan = checked_plan()
    frozen = checked_frozen(plan) if stage != 'development' else None
    if frozen is not None:
        require(frozen['selected_candidate'] != CONTROL, 'No candidate qualified; use --stage finalize without confirmation/auxiliary')
        common.authorize(stage, 'geography_catboost')
    target = OUT / 'evaluation' / stage
    if target.exists():
        return checked_result(stage, plan, frozen)
    sources, inputs = source_hashes(), data_hashes(plan)
    data, X, y, Xq, yq, fi, qi = common.partition(stage)
    contract, weights = split_contract(data), target_weights(data)
    require(data['metadata']['source_sha256'] == plan['original_input_sha256'], 'Original input identity differs from plan')
    planned = plan['split_contract'][stage]
    require(planned['fit_rows'] == len(fi) and planned['query_rows'] == len(qi), 'Planned partition counts differ')
    for label, index in [('fit',fi),('query',qi)]:
        digest = hashlib.sha256(np.asarray(index,dtype='<i8').tobytes()).hexdigest()
        require(planned[label + '_indices_content_sha256'] == digest, 'Planned index contents differ: ' + label)
    if frozen is not None:
        require(contract == frozen['splits'] and weights == frozen['target_group_weights'], 'Frozen indices/group weights differ')
        require(list(X.columns) == frozen['feature_columns'], 'Frozen feature order differs')
    names = list(FORMULAS) if stage == 'development' else [CONTROL,frozen['selected_candidate']]
    reference, provenance = verify_baseline(stage, data, X, Xq, fi, qi, frozen)
    components, parameters = {}, {}
    required = sorted({component for name in names for component in REQUIRED_COMPONENTS[name]})
    for component in required:
        values, params, hashes = verify_component(stage, component, data, X, Xq, fi, qi, common.baseline_folder(stage) / 'reference_v2.pkl', frozen)
        components[component] = values; parameters[component] = params; provenance.update(hashes)
    common.verify_hashes(sources); common.verify_hashes(inputs); common.verify_hashes(provenance)
    folder = common.baseline_folder(stage)
    predictions = build_predictions(names, vector(folder / 'catboost_raw.npy',len(qi)), vector(folder / 'lightgbm_l1_unit63.npy',len(qi)), components, reference, Xq)
    require(np.allclose(predictions[CONTROL], vector(folder / 'v2_control.npy',len(qi)), rtol=1e-12, atol=1e-6), 'Recomputed control differs from persisted baseline')
    # Only the one complete blend and control receive query-label scores; no component selection.
    metrics = {name: score(yq, prediction, X, Xq, common.SPLITS[stage][0], weights) for name,prediction in predictions.items()}
    decision = development_selection(metrics) if stage == 'development' else stage_decision(stage, metrics, frozen['selected_candidate'])
    common.verify_hashes(sources); common.verify_hashes(inputs); common.verify_hashes(provenance)
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = target.parent / ('.' + stage + '-staging-' + uuid.uuid4().hex); staging.mkdir()
    for name,prediction in predictions.items():
        np.save(staging / (name + '.npy'), prediction, allow_pickle=False)
    fields = ['candidate','overall_mae','weighted_mae'] + [group + '_mae' for group in GROUPS] + [q + '_mae' for q in QUARTERS]
    with (staging / 'comparison.csv').open('x',encoding='utf-8-sig',newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for name,value in metrics.items():
            writer.writerow({'candidate':name,'overall_mae':value['overall_mae'],'weighted_mae':value['weighted_mae'],
                **{group+'_mae':value['strata'][group]['mae'] for group in GROUPS},
                **{q+'_mae':value['forecast_quarters'][q]['mae'] for q in QUARTERS}})
    outputs = {rel(target / path.name):sha256(path) for path in staging.iterdir() if path.is_file()}
    report = {'format_version':1,'stage':stage,'created_at_utc':now(),'plan_sha256':sha256(OUT/'candidate_plan.json'),
        'candidate_formulas':FORMULAS,'thresholds':THRESHOLDS,'evaluated_candidates':names,'metrics':metrics,'decision':decision,
        'splits':contract,'feature_columns':list(X.columns),'target_group_weights':weights,'component_parameters':parameters,
        'source_sha256':sources,'data_sha256':inputs,'input_artifact_sha256':provenance,'output_artifact_sha256':outputs,
        'runtime_versions':runtime_versions(),'prediction_units':'TWD','rounded':False,'query_labels_used_for_prediction':False,
        'component_scored_directly':False,'reference_trend_applications_per_candidate':1,
        'geo_feature_rules':FEATURE_RULES,'fit_X_state_reconstruction_verified':True,'query_geographic_state_unchanged':True,'reference_uses_original_features':True,
        'historically_untouched':False,'historical_confirmation_control_previously_evaluated':True,'public_score_used_as_target':False,
        'frozen_candidate_sha256':None if frozen is None else sha256(FREEZE)}
    common.locked_write(staging/'results.json', report)
    require(not target.exists() and staging.parent == target.parent, 'Unsafe result publication or target exists')
    staging.rename(target)  # Leave any failed staging directory for inspection; never recursively delete.
    return checked_result(stage, plan, frozen)


def freeze_development(report):
    plan = checked_plan()
    verified = checked_result('development',plan)
    require(verified == report, 'Development report changed before freeze')
    if FREEZE.exists():
        frozen = checked_frozen(plan)
        require(frozen['selected_candidate'] == report['decision']['selected_candidate'], 'Existing selected candidate differs')
        return frozen
    for path in [OUT/'models/confirmation',OUT/'models/auxiliary',common.baseline_folder('auxiliary')]:
        require(not path.exists() or not any(item.is_file() for item in path.rglob('*')), 'Later-stage fitting artifacts exist before freeze')
    selected = report['decision']['selected_candidate']
    artifacts = {**report['input_artifact_sha256'], **files_hashes(OUT/'evaluation/development')}
    frozen = {'format_version':1,'freeze_timestamp_utc':now(),'selected_candidate':selected,
        'required_components':REQUIRED_COMPONENTS[selected],'plan_sha256':sha256(OUT/'candidate_plan.json'),
        'candidate_formulas':FORMULAS,'thresholds':THRESHOLDS,'geo_feature_rules':FEATURE_RULES,'source_sha256':report['source_sha256'],
        'data_sha256':report['data_sha256'],'development_artifact_sha256':artifacts,
        'component_parameters':report['component_parameters'],'target_group_weights':report['target_group_weights'],
        'splits':report['splits'],'feature_columns':report['feature_columns'],'runtime_versions':report['runtime_versions'],
        'later_stage_new_fitting_artifacts_absent_at_freeze':True,'historically_untouched_confirmation':False}
    common.verify_hashes(artifacts)
    common.locked_write(FREEZE,frozen)
    return checked_frozen(plan)


def finalize():
    plan = checked_plan(); frozen = checked_frozen(plan)
    dev = checked_result('development',plan)
    selected = frozen['selected_candidate']
    require(selected == dev['decision']['selected_candidate'], 'Frozen selection differs from development result')
    evidence = files_hashes(OUT/'evaluation/development')
    evidence.update({rel(FREEZE):sha256(FREEZE),rel(FREEZE.with_suffix('.sha256')):sha256(FREEZE.with_suffix('.sha256'))})
    decisions = {}
    if selected != CONTROL:
        for stage in ['confirmation','auxiliary']:
            report = checked_result(stage,plan,frozen)
            decisions[stage] = report['decision']
            evidence.update(report['input_artifact_sha256']); evidence.update(files_hashes(OUT/'evaluation'/stage))
        accepted = all(value['passed'] for value in decisions.values())
        reason = 'Both fixed gates passed' if accepted else 'At least one fixed gate failed; retain v2'
    else:
        accepted, reason = False, 'No development candidate qualified; confirmation/auxiliary fitting not required'
    result = {'accepted':accepted,'selected_candidate':selected,'frozen_candidate_sha256':sha256(FREEZE),
        'evidence_sha256':evidence,'stage_decisions':decisions,'deployment_decision':selected if accepted else 'retain_v2',
        'reason':reason,'plan_sha256':sha256(OUT/'candidate_plan.json'),'candidate_reselection_performed':False,
        'historically_untouched_confirmation':False}
    common.verify_hashes(evidence)
    if ACCEPTANCE.exists():
        saved = common.read_locked(ACCEPTANCE)
        require({key:value for key,value in saved.items() if key != 'finalized_at_utc'} == result, 'Existing final decision differs')
        common.verify_hashes(saved['evidence_sha256'])
        return saved
    result['finalized_at_utc'] = now(); common.locked_write(ACCEPTANCE,result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage',choices=['development','confirmation','auxiliary','finalize'],required=True)
    parser.add_argument('--freeze',action='store_true')
    args = parser.parse_args()
    require(not args.freeze or args.stage == 'development', '--freeze is only valid for development')
    if args.stage == 'finalize':
        result = finalize()
        summary = {key:result[key] for key in ['accepted','selected_candidate','deployment_decision','reason']}
    else:
        result = evaluate(args.stage)
        if args.freeze:
            freeze_development(result)
        summary = {'stage':args.stage,'decision':result['decision'],'frozen':args.freeze}
    print(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False),flush=True)


if __name__ == '__main__':
    main()
