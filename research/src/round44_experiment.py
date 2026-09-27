"""R44: fit-price weighted log geography; fixed three-branch median, three years."""
from __future__ import annotations
import argparse
import contextlib
import json
import pickle
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
import round39_city_road_experiment as previous
import round44_weighted_log as component

ROOT, OUT = previous.ROOT, previous.ROOT / 'outputs/round44'
DOCUMENT = ROOT / 'reports/round44_plan.md'
STAGES = ('development', 'confirmation', 'auxiliary')
CONTROL, COMPARATOR = 'v9_control', 'v5_log_spatial'
CANDIDATE = 'weighted_log_three_branch_median'
HELPER_SHA = 'e05608cc22d0a1a4c5ce830a374a131ae22b1d1193155b2086151af328695d18'
WEIGHT_FORMULA = '(1 + y_TWD / 1e6) / mean_fit(1 + y_TWD / 1e6)'
AUTHORITIES = {
 'outputs/round6/candidate_plan.json':'606f4d8d8dd0d10175fd11d66318036bd19926beb3931aadc1612326668ecef8',
 'outputs/round39/evaluation/plan.json':'110906f7f284081ba39cc94f6d1389d0432b63d43e724916cc18767392df2b13',
 'outputs/round39/evaluation/results.json':'1912089d214fadbc604d5be519c8a45006166cfa7fc4643afbbc513063578f27',
 'outputs/round36/results.json':'133d99e9fdca21ddf97e58fa8c7860ac74f7311bfe56f0005244c3a76791ef9e',
 'outputs/round40/results.json':'aeb5e2f7d65802ace77d765848725032d89bcd326a3e45a5026627e2cc9cd9b1',
 'outputs/round43/plan.json':'6f42a9f5c5e57c2a25d9c4c5668007c0f824d6827dbaa9b9dd2939c16bff07c5',
 'outputs/round43/results.json':'6b5f0ea14752b9c70b1c6e89df96f4ebd7ac083054518f0d3485451636bd5387',
}
FORMULA = 'median(saved_R39_raw_spatial,new_weighted_log_spatial,saved_R40_unit_spatial)'
BASE_FORMULA = '.75*(.5*old_raw_TWD+.5*new_weighted_log_geo_TWD)+.25*old_unit_LGB_TWD'
need, sha, rel, locked, write = previous.need, previous.sha, previous.rel, previous.locked, previous.write
merge, now, vector, exact, spatial = previous.merge, previous.now, previous.vector, previous.exact, previous.spatial
geo_encoder = component.base


def sources():
    return merge(previous.sources(), {rel(Path(__file__)):sha(Path(__file__)),
        'src/round44_weighted_log.py':sha(ROOT/'src/round44_weighted_log.py'),
        'src/round12_log_geo.py':sha(ROOT/'src/round12_log_geo.py')})


def protected_paths():
    names=['outputs/submission_v2.csv','outputs/submission_v3.csv','runs/versions.json']
    names += [f'outputs/submission_v{v}_trial.csv' for v in range(4,10)]
    return [n for n in names if (ROOT/n).exists()]


def fit_weights(y):
    a=np.asarray(y,dtype='float64')
    need(a.ndim==1 and len(a)>0 and np.isfinite(a).all() and (a>=0).all(),'Invalid fit labels')
    raw=1+a/1e6
    normalization=float(np.mean(raw,dtype='float64'))
    need(np.isfinite(raw).all() and (raw>0).all() and np.isfinite(normalization) and normalization>0,'Invalid fit weight normalization')
    weights=raw/normalization
    need(np.isfinite(weights).all() and (weights>0).all(),'Invalid normalized fit weights')
    return weights,normalization


def freeze():
    need(sha(ROOT/'src/round44_weighted_log.py')==HELPER_SHA,'Fixed helper not ready/changed')
    inputs={rel(DOCUMENT):sha(DOCUMENT)}
    def small(name,expected=None):return previous.prior.small(inputs,ROOT/name,expected)
    def bind(name,digest):inputs.update(merge(inputs,{name:digest}))
    docs={n:small(n,d) for n,d in AUTHORITIES.items()}
    p6=docs['outputs/round6/candidate_plan.json'];p39=docs['outputs/round39/evaluation/plan.json']
    r39=docs['outputs/round39/evaluation/results.json'];r36=docs['outputs/round36/results.json']
    r40=docs['outputs/round40/results.json'];p43=docs['outputs/round43/plan.json'];r43=docs['outputs/round43/results.json']
    need(previous.sources()==p39['source_sha256'] and all(r['checked_stages']==list(STAGES) for r in (r39,r36,r40,r43))
         and r43['plan_sha256']==AUTHORITIES['outputs/round43/plan.json']
         and p43['split_contract']==p39['split_contract'] and p43['target_group_weights']==p39['target_group_weights'],'Prior authority differs')
    for name in ('data_cache/train.pkl','data_cache/train_features.pkl','reports/feature_schema.json',previous.KT):bind(name,p39['input_sha256'][name])
    bind('runs/v1/models/final_config.json',p6['data_sha256']['runs/v1/models/final_config.json'])
    bindings={}
    for stage in STAGES:
        rp=f'outputs/round39/evaluation/{stage}/results.json';r=small(rp,r39['stage_result_sha256'][stage])
        vp=f'outputs/round36/{stage}/results.json';v=small(vp,r36['stage_result_sha256'][stage])
        up=f'outputs/round40/{stage}/results.json';u=small(up,r40['stage_result_sha256'][stage])
        cp=f'outputs/round43/{stage}/results.json';c=small(cp,r43['stage_result_sha256'][stage])
        need(c==r43['stages'][stage] and u==r40['stages'][stage]
             and all(x['stage']==stage and x['status']=='completed' for x in (r,v,u,c))
             and all(x['fit_rows']==p39['split_contract'][stage]['fit_rows'] and x['query_rows']==p39['split_contract'][stage]['query_rows']
                     and x['dates']==p39['splits'][stage] for x in (r,v,u,c)),'Saved window differs')
        need(c['metrics']['v5']==v['metrics']['log_spatial_combination'],'v5 authorities disagree')
        manifest=small(f'outputs/round39/geography/{stage}/manifest.json',r['geo_manifest_sha256'])
        need(manifest==p39['prepared_geography'][stage],'R39 prepared geography differs')
        original=small(f'outputs/round6/geography/{stage}/manifest.json',manifest['original_manifest_sha256'])
        train_geo=original['features']['train']
        need(train_geo['rows']==455255 and p39['input_sha256'][train_geo['path']]==train_geo['sha256'],'Original annual geo differs')
        bind(train_geo['path'],train_geo['sha256'])
        b={'R39_result':rp,'v5_result':vp,'v9_result':cp,'manifest':manifest,'original_geo':train_geo['path'],'reference':r['reference_path']}
        bind(b['reference'],r['reference_sha256'])
        for role,filename in (('fit_indices','fit_indices.npy'),('query_indices','query_indices.npy'),
            ('raw_spatial','new_raw_spatial.npy'),('old_log_spatial','new_log_spatial.npy'),('state','spatial_state.pkl'),('state_summary','state_summary.json')):
            name=f'outputs/round39/evaluation/{stage}/{filename}';b[role]=name;bind(name,r['artifact_sha256'][name])
        receipt=rel((ROOT/b['state_summary']).with_suffix('.sha256'));bind(receipt,r['artifact_sha256'][receipt])
        for role,filename in (('old_raw','catboost_raw.npy'),('old_unit','lightgbm_l1_unit63.npy')):
            name=rel(previous.prior.c6.baseline_folder(stage)/filename);b[role]=name;bind(name,p39['input_sha256'][name])
        b['v5']=f'outputs/round36/{stage}/log_spatial_combination.npy';bind(b['v5'],v['artifact_sha256'][b['v5']])
        b['v9']=f'outputs/round43/{stage}/predictions.npy';bind(b['v9'],c['artifact_sha256'][b['v9']])
        b['unit_spatial']=f'outputs/round40/{stage}/unit_spatial.npy';bind(b['unit_spatial'],u['artifact_sha256'][b['unit_spatial']])
        need(inputs[b['unit_spatial']]==p43['input_sha256'][b['unit_spatial']],'R43 unit input differs')
        b['query_geo']=manifest['features']['query']['path'];bind(b['query_geo'],manifest['features']['query']['sha256']);bindings[stage]=b
    src=sources();previous.prior.c6.verify_hashes(merge(inputs,src))
    params=component.parameters()
    need(params==geo_encoder.parameters() and params['iterations']==2999 and params['depth']==8 and params['learning_rate']==.065
         and params['random_seed']==42 and params['boosting_type']=='Plain' and params['loss_function']=='MAE','R6 recipe differs')
    value={'round':44,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'comparator':COMPARATOR,
        'formula':FORMULA,'base_formula':BASE_FORMULA,'weight_formula':WEIGHT_FORMULA,'source_sha256':src,'input_sha256':inputs,
        'bindings':bindings,'parameters':params,'stage_order':list(STAGES),'splits':p39['splits'],'split_contract':p39['split_contract'],
        'target_group_weights':p39['target_group_weights'],'spatial_parameters':spatial.parameters(),'runtime':previous.average.runtime(),
        'feature_count':95,'categorical_count':23,'new_tree_fits_planned':3,'all_three_periods_regardless_of_scores':True,
        'new_acceptance_thresholds':None,'formal_adoption_decision':None,'automatic_release_allowed':False,
        'fresh_blind_validation':False,'query_labels_passed_to_fit':False,'protected_sha256':{n:sha(ROOT/n) for n in protected_paths()}}
    need(src==sources() and all(sha(ROOT/n)==h for n,h in AUTHORITIES.items()) and sha(DOCUMENT)==inputs[rel(DOCUMENT)],'Freeze authority changed')
    write(OUT/'plan.json',value);print('R44_plan_frozen',sha(OUT/'plan.json'),flush=True)


def native(model, state, cats):
    names = state['all_feature_names']; category_names = list(cats) + geo_encoder.CATEGORICAL
    need(len(names) == 95 and len(category_names) == 23 and list(model.feature_names_) == names
         and model.get_cat_feature_indices() == sorted(names.index(c) for c in category_names), 'Native 95/23 order differs')
    p = model.get_all_params(); leaves = np.asarray(model.get_tree_leaf_counts())
    need(model.tree_count_ == 2999 and p['depth'] == 8 and p['grow_policy'] == 'SymmetricTree'
         and p['boosting_type'] == 'Plain' and p['random_seed'] == 42
         and p['loss_function'] == p['eval_metric'] == 'MAE'
         and np.isfinite(p['learning_rate']) and np.isclose(p['learning_rate'], .065, rtol=1e-7, atol=0)
         and leaves.shape == (2999,) and leaves.dtype.kind in 'iu' and (leaves > 0).all() and (leaves <= 256).all(), 'Native recipe differs')
    return p

def raw_predict(model,state,X,geo,cats):
    component.check_state(state,X,cats)
    frame=geo_encoder.augment_features(X,geo)
    need(list(frame)==state['all_feature_names'],'Model feature order differs')
    native(model,state,cats)
    z=np.asarray(model.predict(Pool(frame,cat_features=list(cats)+geo_encoder.CATEGORICAL,thread_count=4),thread_count=4),dtype='float64')
    need(z.shape==(len(X),) and np.isfinite(z).all(),'Invalid log prediction')
    prices=component.inverse_transform(z)
    need(prices.shape==z.shape and np.isfinite(prices).all() and (prices>=0).all(),'Invalid inverse TWD')
    return z,prices


def evaluate(stage, plan, Xall, yall):
    started = time.monotonic(); b = plan['bindings'][stage]; folder = OUT / stage
    folder.mkdir(exist_ok=False); model_folder = folder / 'model'; model_folder.mkdir()
    fi, qi = [np.load(ROOT / b[k + '_indices'], allow_pickle=False) for k in ('fit', 'query')]
    spec = plan['split_contract'][stage]
    for name, a in (('fit', fi), ('query', qi)):
        need(a.dtype == np.dtype('int64') and a.shape == (spec[name + '_rows'],)
             and (np.diff(a) > 0).all() and a.min() >= 0 and a.max() < len(Xall)
             and previous.prior.log_run.original.value_sha(a, '<i8') == spec[name + '_indices_content_sha256'], 'Canonical indices differ')
        previous.save_array(model_folder / (name + '_indices.npy'), a)
    need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
    X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
    y, yq = yall[fi], yall[qi]
    need(y.shape == (len(X),) and np.isfinite(y).all() and (y >= 0).all(), 'Invalid fit target')
    allgeo = pd.read_pickle(ROOT / b['original_geo']); geo_encoder.check_geo(allgeo)
    need(len(allgeo) == len(Xall) and allgeo.index.equals(Xall.index), 'Canonical original geo differs')
    geo, geoq = allgeo.iloc[fi].reset_index(drop=True), allgeo.iloc[qi].reset_index(drop=True)
    del allgeo
    newgeo = pd.read_pickle(ROOT / b['query_geo']); geo_encoder.check_geo(newgeo)
    need(newgeo.index.equals(Xq.index) and spatial.frame_sha(newgeo) == b['manifest']['features']['query']['frame_sha256'], 'New spatial query geo differs')
    before = {k: spatial.frame_sha(v) for k, v in {'fit_X': X, 'query_X': Xq, 'original_fit_geo': geo, 'original_query_geo': geoq}.items()}
    old_result, v5_result, v9_result = [locked(ROOT / b[k]) for k in ('R39_result','v5_result','v9_result')]
    need(all(old_result['input_frame_sha256'][k] == v for k, v in before.items()), 'Original X/geo identity differs')
    with (ROOT / b['state']).open('rb') as f: spatial_state = pickle.load(f)
    summary = locked(ROOT / b['state_summary'])
    with (ROOT / b['reference']).open('rb') as f: reference = pickle.load(f)
    need(spatial_state['state_summary'] == summary and spatial_state['parameters'] == plan['spatial_parameters']
         and summary['fit_X_sha256'] == before['fit_X'] and summary['query_used_in_build'] is False
         and summary['fit_geo_sha256'] == b['manifest']['features']['fit']['frame_sha256']
         and reference['fit_report']['fitting_rows'] == len(X)
         and summary['existing_reference_residual_sha256'] == previous.prior.log_run.original.value_sha(reference['reference_residual'], '<f8'), 'Fit-only spatial/reference identity differs')
    raw, old_log, unit_spatial, control, v5 = [vector(ROOT/b[k],len(qi)) for k in ('raw_spatial','old_log_spatial','unit_spatial','v9','v5')]
    exact(np.median(np.column_stack([raw,old_log,unit_spatial]),axis=1),control,'v9 saved branch median')
    score = lambda p: previous.prior.log_run.original.score(yq, p, X, Xq, plan['splits'][stage][0], plan['target_group_weights'])
    metrics = {CONTROL: score(control), COMPARATOR: score(v5)}
    need(metrics[CONTROL] == v9_result['metrics']['three_branch_median']
         and metrics[COMPARATOR] == v5_result['metrics']['log_spatial_combination'], 'Saved full control metrics differ')
    cats = previous.prior.c6.read(ROOT / 'reports/feature_schema.json')['categorical_columns']
    target_sha = previous.prior.log_run.original.value_sha(y, '<f8')
    expected_weights, weight_mean = fit_weights(y)
    weight_sha = component.values_sha256(expected_weights)
    write(folder / 'fit_started.json', {'round': 44, 'stage': stage, 'started_at_utc': now(),
        'plan_sha256': sha(OUT / 'plan.json'), 'fit_rows': len(X), 'query_rows': len(Xq),
        'parameters': plan['parameters'], 'fit_target_sha256': target_sha, 'fit_weights_sha256': weight_sha, 'weight_formula':WEIGHT_FORMULA, 'query_labels_used': False, 'automatic_retry': False})
    print(json.dumps({'event': 'fit_started', 'stage': stage, 'fit_rows': len(X), 'query_rows': len(Xq)}), flush=True)
    tick = time.monotonic(); model, state, metadata = component.fit_model(X, y, geo, cats)
    fit_seconds = time.monotonic() - tick
    model.save_model(str(model_folder / 'model.cbm')); write(model_folder / 'feature_state.json', state)
    fitted_native = native(model, state, cats)
    need(metadata['parameters']==plan['parameters'] and metadata['feature_count']==95 and metadata['categorical_feature_count']==23
         and metadata['weight_argument_passed_to_pool'] is True and metadata['weight_formula']==WEIGHT_FORMULA
         and metadata['fit_weights_sha256']==weight_sha and metadata['fit_original_target_values_sha256']==target_sha
         and metadata['fit_transformed_target_values_sha256']==component.values_sha256(component.transform_target(y))
         and metadata['fit_weight_summary']['normalization_mean_fit_only']==weight_mean
         and state['geo_state']['fit_original_feature_values_sha256']==before['fit_X'] and state['fit_rows']==len(X)
         and state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False
         and not metadata['fit_uses_eval_set'] and not metadata['early_stopping_used'],'Weighted log component contract differs')
    metadata.update(round=44, stage=stage, plan_sha256=sha(OUT / 'plan.json'), source_sha256=plan['source_sha256'],
        fitted_native_parameters=fitted_native, fit_elapsed_seconds=fit_seconds,
        original_frame_sha256=before, query_labels_passed_to_fit=False, prediction_units='TWD')
    write(model_folder / 'fit_metadata.json', metadata)
    z, prices = raw_predict(model, state, Xq, geoq, cats)
    previous.save_array(model_folder / 'raw_z.npy', z); previous.save_array(model_folder / 'predictions.npy', prices)
    del model
    restored = CatBoostRegressor(); restored.load_model(str(model_folder / 'model.cbm'))
    saved_state = locked(model_folder / 'feature_state.json'); need(saved_state == state, 'State JSON roundtrip differs')
    rz, rp = raw_predict(restored, saved_state, Xq, geoq, cats)
    exact(z, rz, 'Full raw saved model replay'); exact(prices, rp, 'Full TWD saved model replay')
    metadata.update(saved_native_parameters=native(restored, saved_state, cats),
        raw_z_diagnostics={'rows': len(z), 'negative_count': int((z < 0).sum()), 'zero_count': int((z == 0).sum()),
                           'min': float(z.min()), 'max': float(z.max()), 'units': 'log1p_TWD_per_million'},
        reload_rows=len(Xq), raw_TWD_reload_bitwise_equal=True,
        artifact_sha256={rel(p): sha(p) for p in sorted(model_folder.iterdir()) if p.is_file()})
    del restored
    need(previous.prior.log_run.original.value_sha(y, '<f8') == target_sha, 'Fit target mutated')
    write(model_folder / 'metadata.json', metadata)
    old, unit = vector(ROOT / b['old_raw'], len(qi)), vector(ROOT / b['old_unit'], len(qi))
    base = .75 * (.5 * old + .5 * prices) + .25 * unit
    p, diag, reference_report = previous.prior.log_run.original.predict_reference(reference, Xq, base, variant='trend', return_diagnostics=True)
    branch, detail, spatial_report = spatial.predict(spatial_state, Xq, newgeo, base, p, diag)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    exact(branch[fallback], p[fallback], 'Weighted log spatial fallback')
    candidate = np.median(np.column_stack([raw,branch,unit_spatial]),axis=1)
    need(np.isfinite(candidate).all() and (candidate >= 0).all(), 'Invalid candidate TWD')
    metrics[CANDIDATE] = score(candidate)
    for name, values in ((CONTROL, control), (COMPARATOR, v5), ('base_weighted_log', base), ('reference_weighted_log', p),
                         ('weighted_log_spatial', branch), (CANDIDATE, candidate)):
        previous.save_array(folder / (name + '.npy'), values)
    for name, value in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
        with (folder / (name + '.pkl')).open('xb') as f: pickle.dump(value, f, protocol=5)
    need(before == {k: spatial.frame_sha(v) for k, v in {'fit_X': X, 'query_X': Xq, 'original_fit_geo': geo, 'original_query_geo': geoq}.items()}
         and spatial.frame_sha(newgeo) == b['manifest']['features']['query']['frame_sha256'], 'Original/new frames mutated')
    result = {'round': 44, 'stage': stage, 'status': 'completed', 'created_at_utc': now(),
        'plan_sha256': sha(OUT / 'plan.json'), 'fit_rows': len(fi), 'query_rows': len(qi), 'dates': plan['splits'][stage],
        'metrics': metrics, 'candidate_vs_v9': previous.prior.log_run.differences(metrics[CONTROL], metrics[CANDIDATE]),
        'candidate_vs_v5': previous.prior.log_run.differences(metrics[COMPARATOR], metrics[CANDIDATE]),
        'v9_control_bitwise_equal': True, 'v9_and_v5_all_metrics_exact': True,
        'fit_weight_summary': metadata['fit_weight_summary'], 'fit_weights_sha256': weight_sha, 'weight_formula': WEIGHT_FORMULA,
        'original_X_used_for_reference_spatial_and_score': True, 'original_frames_unchanged': True,
        'original_geo_used_for_tree': True, 'new_geo_used_only_for_spatial': True,
        'new_tree_fits': 1, 'reference_predict_calls': 1, 'new_spatial_predict_calls': 1, 'new_spatial_state_builds': 0,
        'query_labels_used_for_fit_or_predict': False, 'full_query_raw_TWD_reload_bitwise_equal': True,
        'reference_report': reference_report, 'spatial_report': spatial_report,
        'model_metadata_sha256': sha(model_folder / 'metadata.json'), 'fit_elapsed_seconds': fit_seconds,
        'elapsed_seconds': time.monotonic() - started, 'formal_adoption_decision': None,
        'new_acceptance_thresholds': None, 'new_submission_generated': False, 'fresh_blind_validation': False,
        'artifact_sha256': {rel(p): sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}}
    write(folder / 'results.json', result)
    print(json.dumps({'event': 'stage_complete', 'stage': stage, 'vs_v9': result['candidate_vs_v9'], 'vs_v5': result['candidate_vs_v5']}), flush=True)
    return result

def run():
    p = locked(OUT / 'plan.json')
    need(p['round'] == 44 and p['stage_order'] == list(STAGES) and p['all_three_periods_regardless_of_scores'] is True
         and p['parameters'] == component.parameters() and p['weight_formula'] == WEIGHT_FORMULA
         and p['source_sha256'] == sources()
         and p['runtime'] == previous.average.runtime() and p['spatial_parameters'] == spatial.parameters()
         and p['formula'] == FORMULA and p['base_formula'] == BASE_FORMULA
         and p['new_acceptance_thresholds'] is None and p['automatic_release_allowed'] is False, 'Frozen method changed')
    need(not (OUT / 'run_claim.json').exists() and not (OUT / 'results.json').exists()
         and not any((OUT / s).exists() for s in STAGES), 'Prior attempt exists; preserve it')
    write(OUT / 'run_claim.json', {'started_at_utc': now(), 'plan_sha256': sha(OUT / 'plan.json'), 'automatic_retry': False})
    result = {'round': 44, 'status': 'incomplete', 'plan_sha256': sha(OUT / 'plan.json'), 'stage_order': list(STAGES),
        'checked_stages': [], 'stages': {}, 'stage_result_sha256': {}, 'source_sha256': p['source_sha256'],
        'input_sha256': p['input_sha256'], 'protected_before': p['protected_sha256'],
        'formal_adoption_decision': None, 'new_acceptance_thresholds': None, 'new_submission_generated': False,
        'registry_updated': False, 'fresh_blind_validation': False,
        'label_loading': 'shared loader deserializes canonical train targets; fit y only to tree, query y only to scorer'}
    start = time.monotonic(); phase = 'verify_inputs_before_load'
    try:
        previous.prior.c6.verify_hashes(merge(p['input_sha256'], p['source_sha256'], p['protected_sha256']))
        Xall, yall = previous.load_training_rows(p)
        for stage in STAGES:  # No branch depends on earlier scores.
            phase = stage; item = evaluate(stage, p, Xall, yall)
            result['checked_stages'].append(stage); result['stages'][stage] = item
            result['stage_result_sha256'][stage] = sha(OUT / stage / 'results.json')
        previous.prior.c6.verify_hashes(merge(p['source_sha256'], p['protected_sha256']))
        result['status'] = 'three_periods_completed_no_adoption_decision'
    except Exception:
        result.update(status='incomplete_technical_failure', failure_phase=phase, traceback=traceback.format_exc())
        print(result['traceback'], flush=True)
    result.update(elapsed_seconds=time.monotonic() - start, completed_at_utc=now(),
        fit_started_stages=[s for s in STAGES if (OUT / s / 'fit_started.json').exists()],
        model_saved_stages=[s for s in STAGES if (OUT / s / 'model/model.cbm').exists()],
        protected_after={n: sha(ROOT / n) for n in p['protected_sha256']})
    write(OUT / 'results.json', result)
    return 0 if result['status'] == 'three_periods_completed_no_adoption_decision' else 1

def execute(mode):
    if mode == 'freeze':
        need(not OUT.exists(), 'R44 output already exists'); OUT.mkdir(parents=True, exist_ok=False)
    else: need(OUT.is_dir(), 'Freeze is required')
    log = OUT / (mode + '.log'); started = now(); tick = time.monotonic(); code = 1; failure = None
    with log.open('x', encoding='utf-8') as f:
        with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
            try:
                if mode == 'freeze': freeze(); code = 0
                else: code = run()
            except Exception:
                failure = traceback.format_exc(); print(failure, flush=True)
    record = {'round': 44, 'command': sys.argv, 'started_at_utc': started, 'completed_at_utc': now(),
        'elapsed_seconds': time.monotonic() - tick, 'exit_code': code, 'failure': failure,
        'source_sha256': sha(Path(__file__)), 'stdout_log': rel(log), 'stdout_log_sha256': sha(log), 'automatic_retry': False}
    write(OUT / (mode + '_execution.json'), record)
    print(json.dumps(record, ensure_ascii=False), flush=True)
    return code

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('command',choices=['freeze','run'])
    raise SystemExit(execute(parser.parse_args().command))
