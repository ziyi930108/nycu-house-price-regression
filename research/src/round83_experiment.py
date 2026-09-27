"""R83: one seed137 unit-CatBoost fit per period; fixed mean with saved R79 seed42."""
from pathlib import Path
import argparse
import contextlib
import json
import pickle
import sys
import time
import traceback
import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import round40_three_branch as unit_run
import round48_nn_experiment as old
import round79_recency_unit as baseline
import round83_seed_unit as helper

ROOT, STAGES = old.ROOT, old.STAGES
OUT, DOCUMENT = ROOT / 'outputs/round83', ROOT / 'reports/round83_plan.md'
PLAN_SHA = 'c0cf070337ca86ab742479b664628413155f1e69745c7a80ae1f15e152c747c0'
HELPER_SHA = 'c8d39865c161b73b06f54ca9cc5858b85864b149f15ae81c6e2868797a3b0a8b'
SELECTOR_SHA = 'c993a861fa6766fb5c9da8c156521bb28886a459587653ee8734b2927b9e87a8'
AUTHORITIES = {'outputs/round79/results.json': '7b50e0a236f84649e20e5e35e72b2c7d09259322063dd0e0b2464cdba3240df6',
    'outputs/round80/results.json': 'dc55eac2f930301edf2efd78c778f56edc0d96c37d4022c2cc25f22a07dfe9e6',
    'outputs/round79/execution.json': '8b0132b6ad0b374488f02957661fdc1b46ac69c381c0e71717498c02e18f986d',
    'outputs/round80/execution.json': '973fd036c1b9a107a067918be4686a3deb9d7399292974ce35b74f295723b577'}
BACKGROUND = ('v22', 'v20', 'v21', 'v19', 'v18', 'v11', 'v12', 'v5')
CONTROLS = ('v23', *BACKGROUND)
FORMULA = 'median(saved_R71_seed42_nn_spatial,saved_R46_log_spatial,spatial(reference_trend(.5*saved_R79_seed42_TWD+.5*new_seed137_TWD)))'
NPZ_KEYS = {'train_num', 'query_num', 'train_cat', 'query_cat', 'train_y_million', 'cardinalities',
    'train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'}
sha, read, write, spatial, rel = old.sha, old.locked, old.write, old.spatial, old.rel


def exact(a, b, label):
    assert a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes(), label


def prices(v, n):
    assert v.dtype == np.float64 and v.shape == (n,) and np.isfinite(v).all() and (v >= 0).all()
    return v


def weight_summary(values):
    v = np.asarray(values, dtype=np.float64)
    total, squares = float(v.sum(dtype=np.float64)), float(np.dot(v, v))
    return dict(rows=len(v), min=float(v.min()), max=float(v.max()), mean=float(v.mean()),
        sum=total, sum_squares=squares, effective_sample_size=total*total/squares)


def summary_equal(a, b):
    for key in ('rows', 'min', 'max'):
        assert a[key] == b[key], key
    for key in ('mean', 'sum', 'sum_squares', 'effective_sample_size'):
        assert np.isclose(a[key], b[key], rtol=1e-12, atol=1e-12), key


def date_weights(binding, X, yfit, n):
    prep, meta = binding['preparation'], binding['R79_metadata']
    with np.load(ROOT / binding['prepared'], allow_pickle=False) as z:
        assert set(z.files) == NPZ_KEYS
        w = z['train_sample_weight'].copy()
        exact(z['train_y_million'], np.asarray(yfit/1e6, dtype=np.float32), 'Saved fit labels')
    assert w.dtype == np.float32 and w.shape == (len(X),) and np.isfinite(w).all() and (w > 0).all()
    assert old.original.value_sha(w, '<f4') == prep['train_sample_weight_sha256'] == meta['fit_date_weights_sha256']
    for saved in (prep['train_sample_weight_summary'], binding['R63_training']['train_sample_weight_summary'], meta['fit_date_weights_summary']):
        summary_equal(weight_summary(w), saved)
    assert prep['fit_rows'] == len(X) and prep['query_rows'] == n
    dates = prep['date_weight_summary_float64']
    d, invalid, outside = (X[k].to_numpy(dtype=np.float64) for k in ('trade_day_index', 'trade_date_invalid', 'trade_date_out_of_range'))
    for v, key in ((d, 'date_values_sha256'), (invalid, 'invalid_flags_sha256'), (outside, 'out_of_range_flags_sha256')):
        assert baseline.values_sha256(v) == dates[key]
    valid = np.isfinite(d) & (invalid == 0) & (outside == 0)
    assert dates['reference_rows'] == len(X) and dates['valid_date_rows'] == int(valid.sum())
    assert dates['neutral_date_rows'] == int((~valid).sum()) and (w[~valid] == np.float32(1)).all()
    assert dates['weight_rules']['half_life_days'] == 365 and not dates['weight_rules']['weights_use_query'] and not dates['weight_rules']['weights_use_targets']
    return w


def fit_native(folder, stage, X, yfit, G, Xq, Gq, cats, w, binding, frames, spec, plan_sha, inputs, runtime):
    model_dir = folder / 'model'
    model_dir.mkdir()
    started, tick, failure = old.now(), time.monotonic(), None
    write(model_dir / 'fit_started.json', dict(started_at_utc=started, helper_sha256=HELPER_SHA,
        parameters=helper.parameters(), prepared_path=binding['prepared'], prepared_sha256=inputs[binding['prepared']],
        date_weight_sha256=binding['preparation']['train_sample_weight_sha256'], query_labels_used=False, automatic_retry=False))
    try:
        with (model_dir / 'fit.log').open('x', encoding='utf8') as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            model, state, metadata = helper.fit_model(X, yfit, G, cats, w)
        fit_seconds = time.monotonic()-tick
        # Persist completed fit before any query prediction or later metadata checks.
        model.save_model(str(model_dir / 'model.cbm'), format='cbm')
        write(model_dir / 'feature_state.json', state)
        write(model_dir / 'fit_metadata.json', metadata)
    except Exception:
        failure = traceback.format_exc()
        raise
    finally:
        seconds = time.monotonic()-tick
        write(model_dir / 'fit_execution.json', dict(exit_code=1 if failure else 0, failure=failure,
            started_at_utc=started, completed_at_utc=old.now(), elapsed_seconds=seconds, automatic_retry=False))
    legacy, before = binding['R79_state'], binding['R79_metadata']
    assert state['legacy_feature_state'] == legacy and metadata['parameters'] == helper.parameters()
    assert metadata['parameters'] == {**before['parameters'], 'random_seed': 137}
    assert metadata['seed'] == 137 and metadata['prediction_units'] == 'TWD'
    for key in ('fit_target_values_sha256', 'fit_transformed_target_values_sha256', 'fit_effective_area_sha256',
                'fit_factor_sha256', 'fit_area_weights_sha256', 'fit_date_weights_sha256', 'fit_sample_weights_sha256',
                'fit_original_feature_values_sha256', 'fit_geo_feature_values_sha256', 'fit_model_feature_values_sha256'):
        assert metadata[key] == before[key], key
    actual_weights = baseline.original.sample_weights(X) * w.astype(np.float64)
    assert metadata['fit_sample_weights_sha256'] == baseline.values_sha256(actual_weights)
    summary_equal(metadata['fit_date_weights_summary'], before['fit_date_weights_summary'])
    summary_equal(metadata['fit_sample_weights_summary'], before['fit_sample_weights_summary'])
    fitted = helper.check_model(model, state)
    raw = helper.predict_raw(model, state, Xq, Gq, cats)
    native = prices(helper.inverse_transform(raw, Xq), len(Xq))
    loaded_state = read(model_dir / 'feature_state.json')
    assert loaded_state == state
    loaded = CatBoostRegressor()
    loaded.load_model(str(model_dir / 'model.cbm'), format='cbm')
    capacity = helper.check_model(loaded, loaded_state)
    for key in fitted:
        if key != 'actual_learning_rate':
            assert capacity[key] == fitted[key], key
    assert fitted['actual_random_seed'] == capacity['actual_random_seed'] == 137
    reload_raw = helper.predict_raw(loaded, loaded_state, Xq, Gq, cats)
    exact(raw, reload_raw, 'All query raw unit reload')
    exact(native, helper.inverse_transform(reload_raw, Xq), 'All query TWD reload')
    for name, value in (('raw_z', raw), ('native_predictions', native), ('date_weights', w), ('pool_weights', actual_weights)):
        old.previous.save_array(model_dir / (name + '.npy'), value)
    del model, loaded
    metadata.update(round=83, stage=stage, query_rows=len(Xq), fit_seconds=fit_seconds, fit_and_save_seconds=seconds,
        fitted_native_capacity=fitted, loaded_native_capacity=capacity, **capacity,
        learning_rate_tolerance=dict(reference=.065, rtol=1e-7, atol=0, fit_reload_exact_required=False),
        original_R79_legacy_feature_state_exact=True, original_frame_sha256=frames,
        date_weight_provenance_verified_in_driver=True, R63_preparation=binding['preparation'],
        fit_indices_content_sha256=spec['fit_indices_content_sha256'], query_indices_content_sha256=spec['query_indices_content_sha256'],
        reload_all_query_rows_verified=True, reload_raw_exact=True, reload_TWD_exact=True, reload_rows=len(Xq),
        native_negative_raw_rows=int((raw < 0).sum()), native_zero_TWD_rows=int((native == 0).sum()),
        runtime=runtime, plan_sha256=plan_sha, input_sha256=dict(inputs))
    metadata['artifact_sha256'] = {rel(p): sha(p) for p in model_dir.iterdir() if p.is_file()}
    write(model_dir / 'metadata.json', metadata)
    return native, fit_seconds


def run(expected_plan_sha):
    tick = time.monotonic()
    assert expected_plan_sha == PLAN_SHA == sha(DOCUMENT) and sha(Path(helper.__file__)) == HELPER_SHA
    inputs = {}
    def checked(name, digest):
        name = name.replace('\\', '/')
        path = (ROOT / name).resolve()
        assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs:
            assert sha(path) == digest, name
        inputs[name] = digest
        return path
    def small(name, digest):
        path = checked(name, digest)
        value = read(path)
        checked(rel(path.with_suffix('.sha256')), sha(path.with_suffix('.sha256')))
        return value
    def artifact(record, name):
        return checked(name, {k.replace('\\', '/'): v for k, v in record['artifact_sha256'].items()}[name])
    checked(rel(DOCUMENT), PLAN_SHA)
    r79, r80 = (small(f'outputs/round{n}/results.json', AUTHORITIES[f'outputs/round{n}/results.json']) for n in (79, 80))
    p79, p80 = (small(f'outputs/round{n}/plan.json', r['plan_sha256']) for n, r in ((79, r79), (80, r80)))
    for n, record, prior_plan in ((79, r79, p79), (80, r80, p80)):
        e = small(f'outputs/round{n}/execution.json', AUTHORITIES[f'outputs/round{n}/execution.json'])
        assert e['exit_code'] == 0 and e['failure'] is None and e['source_sha256'] in record['source_sha256'].values()
        assert record['status'] == 'three_periods_completed_no_adoption_decision' and record['checked_stages'] == list(STAGES)
        assert record['source_sha256'] == prior_plan['source_sha256'] and record['input_sha256'] == prior_plan['input_sha256']
    p48, p40 = (small(f'outputs/round{n}/plan.json', p79['input_sha256'][f'outputs/round{n}/plan.json']) for n in (48, 40))
    assert p79['split_contract'] == p80['split_contract'] == p48['split_contract'] == p40['split_contract']
    assert p79['splits'] == p80['splits'] == p48['splits'] == p40['splits']
    assert p79['target_group_weights'] == p80['target_group_weights'] == p48['target_group_weights'] == p40['target_group_weights']
    assert p40['spatial_parameters'] == spatial.parameters()
    assert helper.parameters() == {**baseline.parameters(), 'random_seed': 137}
    source = old.previous.merge(r79['source_sha256'], r80['source_sha256'], helper.source_hashes(), {rel(Path(__file__)): sha(Path(__file__))})
    for name, digest in source.items():
        checked(name, digest)
    config = 'runs/v1/models/final_config.json'
    checked(config, p79['input_sha256'][config])
    protected = {rel(p): sha(p) for p in sorted((ROOT / 'outputs').glob('submission_v*.csv'))}
    assert len(protected) == 24
    protected['runs/versions.json'] = sha(ROOT / 'runs/versions.json')
    selector_path = ROOT / 'outputs/current_private_candidate.json'
    assert sha(selector_path) == SELECTOR_SHA
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT):
        assert p79['input_sha256'][name] == p48['input_sha256'][name]
        checked(name, p79['input_sha256'][name])
    bindings = {}
    for s in STAGES:
        a, a80, inherited = r79['stages'][s], r80['stages'][s], p79['bindings'][s]
        b, spec = inherited['unit'], p79['split_contract'][s]
        assert a['full_query_raw_TWD_reload_exact'] and a['query_labels_used_only_for_evaluation'] and a['controls_full_metrics_exact']
        assert a80['controls_full_metrics_exact'] and a80['inherited_R79_model_reload_evidence_verified']
        for key in ('fit_indices', 'query_indices', 'reference', 'state', 'state_summary', 'query_geo'):
            checked(b[key], p79['input_sha256'][b[key]])
        receipt = rel((ROOT / b['state_summary']).with_suffix('.sha256'))
        checked(receipt, p79['input_sha256'][receipt])
        for key in ('tree_geo', 'tree_manifest', 'prepared'):
            checked(inherited[key], p79['input_sha256'][inherited[key]])
        gm = small(inherited['tree_manifest'], inputs[inherited['tree_manifest']])
        assert gm['stage'] == s and not gm['geographic_values_learned_from_targets']
        assert inputs[inherited['tree_manifest']] == inherited['R19_metadata']['geo_manifest_sha256'] == b['manifest']['original_manifest_sha256']
        prep_path, training_path = f'outputs/round63/{s}/preparation.json', f'outputs/round63/{s}/model/training.json'
        prep = small(prep_path, p79['input_sha256'][prep_path])
        checked(training_path, p79['input_sha256'][training_path])
        training = json.loads((ROOT / training_path).read_text(encoding='utf8'))
        assert prep == inherited['preparation'] and training == inherited['R63_training']
        assert prep['prepared_path'] == inherited['prepared'] and prep['prepared_sha256'] == inputs[inherited['prepared']]
        assert training['status'] == 'completed' and training['input_sha256'] == inputs[inherited['prepared']]
        assert training['train_sample_weight_sha256'] == prep['train_sample_weight_sha256']
        assert prep['weights_use_fit_X_only'] and not prep['weights_use_targets'] and not prep['weights_use_query_dates']
        assert not prep['query_labels_used'] and not prep['float32_weights_renormalized'] and prep['neutral_weights_exactly_one']
        mp, sp = f'outputs/round79/{s}/model/metadata.json', f'outputs/round79/{s}/model/feature_state.json'
        artifact(a, mp); artifact(a, sp)
        m79, legacy = small(mp, inputs[mp]), small(sp, inputs[sp])
        assert legacy == inherited['R19_state'] and m79['feature_state_content_sha256'] == legacy['state_content_sha256']
        assert m79['parameters'] == baseline.parameters() and m79['prediction_units'] == 'TWD' and m79['seed'] == 42
        assert m79['reload_all_query_rows_verified'] and m79['reload_raw_exact'] and m79['reload_TWD_exact']
        paths = {k: inherited['paths'][k] for k in ('nn', 'log', *BACKGROUND[1:])}
        for name in paths.values():
            checked(name, p79['input_sha256'][name])
        for key, number, suffix in (('seed42', 79, 'model/native_predictions.npy'), ('v22', 79, 'predictions.npy'),
            ('v23', 80, 'predictions.npy'), ('control_base', 80, 'candidate_base.npy'),
            ('control_reference', 80, 'candidate_reference.npy'), ('control_spatial', 80, 'candidate_unit_spatial.npy')):
            paths[key] = f'outputs/round{number}/{s}/{suffix}'
            artifact(a if number == 79 else a80, paths[key])
        assert inputs[paths['seed42']] == m79['artifact_sha256'][paths['seed42']]
        assert inputs[paths['v22']] == a['prediction_sha256'] and inputs[paths['v23']] == a80['prediction_sha256']
        for kind in ('fit', 'query'):
            for number, record in ((79, a), (80, a80)):
                path = f'outputs/round{number}/{s}/{kind}_indices.npy'
                artifact(record, path)
                assert inputs[path] == inputs[b[kind + '_indices']]
            path = f'outputs/round63/{s}/{kind}_indices.npy'
            checked(path, p79['input_sha256'][path])
            assert inputs[path] == inputs[b[kind + '_indices']] and m79[kind+'_indices_content_sha256'] == spec[kind+'_indices_content_sha256']
        bindings[s] = dict(unit=b, paths=paths, tree_geo=inherited['tree_geo'], tree_manifest=inherited['tree_manifest'],
            prepared=inherited['prepared'], preparation=prep, R63_training=training, R79_metadata=m79,
            R79_state=legacy, R19_metadata=inherited['R19_metadata'])
    plan = dict(round=83, candidate='R83', primary_control='v23', secondary_control='v22', text_plan_sha256=PLAN_SHA,
        formula=FORMULA, source_sha256=source, input_sha256=dict(inputs), bindings=bindings, parameters=helper.parameters(),
        baseline_parameters=baseline.parameters(), changed_parameter_names=['random_seed'], seeds=[42,137], reused_seeds=[42], trained_seeds=[137],
        stage_order=list(STAGES), split_contract=p79['split_contract'], splits=p79['splits'], target_group_weights=p79['target_group_weights'],
        planned_new_fits=3, new_NN_fits=0, new_reference_fits=0, new_spatial_state_builds=0,
        all_periods_regardless_of_scores=True, saved_date_weights_recomputed=False, individual_seed_selection=False,
        protected_before=protected, selector_sha256_before=SELECTOR_SHA, query_labels_used_only_for_evaluation=True,
        new_CSV=False, new_test_predictions=False, automatic_adoption=False, fresh_blind_validation=False, Public_feedback_used_for_tuning=False)
    write(OUT / 'plan.json', plan)
    cats = json.loads((ROOT/'reports/feature_schema.json').read_text(encoding='utf8'))['categorical_columns']
    Xall, yall = old.previous.load_training_rows(p48)
    assert len(Xall) == 455255
    result = {k: plan[k] for k in ('round','candidate','primary_control','secondary_control','formula','text_plan_sha256','source_sha256','input_sha256',
        'seeds','reused_seeds','trained_seeds','new_NN_fits','new_reference_fits','new_spatial_state_builds','protected_before','selector_sha256_before',
        'query_labels_used_only_for_evaluation','new_CSV','new_test_predictions','automatic_adoption','fresh_blind_validation','Public_feedback_used_for_tuning','individual_seed_selection')}
    result.update(status='running', checked_stages=[], stages={}, plan_sha256=sha(OUT/'plan.json'), new_fits=0,
        new_submission_generated=False, runtime=dict(catboost=catboost.__version__, numpy=np.__version__, pandas=pd.__version__))
    for s in STAGES:
        start = time.monotonic(); folder = OUT / s; folder.mkdir()
        binding, a, a80, spec = bindings[s], r79['stages'][s], r80['stages'][s], p79['split_contract'][s]
        b, paths, indices = binding['unit'], binding['paths'], {}
        for kind in ('fit','query'):
            ind = np.load(ROOT/b[kind+'_indices'], allow_pickle=False)
            assert ind.dtype == np.int64 and ind.shape == (spec[kind+'_rows'],) and (np.diff(ind)>0).all() and ind.min()>=0 and ind.max()<len(Xall)
            assert old.original.value_sha(ind,'<i8') == spec[kind+'_indices_content_sha256']
            indices[kind] = ind; old.previous.save_array(folder/(kind+'_indices.npy'),ind)
        fi, qi = indices['fit'], indices['query']; assert not np.intersect1d(fi,qi).size
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        yfit, n = yall[fi].copy(), len(qi)
        frames = dict(fit_X=spatial.frame_sha(X), query_X=spatial.frame_sha(Xq))
        assert frames == a['original_frame_sha256'] == a80['original_frame_sha256'] == binding['R79_metadata']['original_frame_sha256']
        m19 = binding['R19_metadata']
        assert all(m19[k+'_sha256'] == v for k,v in frames.items())
        original_geo = pd.read_pickle(ROOT/binding['tree_geo'])
        assert len(original_geo) == len(Xall) and original_geo.index.equals(Xall.index)
        G, Gq = original_geo.iloc[fi].reset_index(drop=True), original_geo.iloc[qi].reset_index(drop=True); del original_geo
        assert spatial.frame_sha(G) == m19['fit_geo_sha256'] and spatial.frame_sha(Gq) == m19['query_geo_sha256']
        assert baseline.fit_feature_state(X,G,cats) == binding['R79_state']
        assert baseline.values_sha256(yfit) == m19['fit_target_values_sha256'] == binding['R79_metadata']['fit_target_values_sha256']
        w = date_weights(binding,X,yfit,n)
        geo = pd.read_pickle(ROOT/b['query_geo'])  # R39: spatial only; trees receive original R6 G/Gq.
        unit_run.previous.geo_encoder.check_geo(geo)
        assert geo.index.equals(Xq.index) and spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256']
        with (ROOT/b['state']).open('rb') as f: spatial_state=pickle.load(f)
        ss=read(ROOT/b['state_summary'])
        assert spatial_state['state_summary']==ss and spatial_state['parameters']==p40['spatial_parameters']
        assert ss['fit_X_sha256']==frames['fit_X'] and ss['fit_geo_sha256']==b['manifest']['features']['fit']['frame_sha256'] and not ss['query_used_in_build']
        with (ROOT/b['reference']).open('rb') as f: reference=pickle.load(f)
        assert reference['fit_report']['fitting_rows']==len(fi) and baseline.values_sha256(reference['reference_residual'])==ss['existing_reference_residual_sha256']
        vector=lambda name:unit_run.vector(ROOT/name,n)
        fixed={k:vector(paths[k]) for k in ('seed42','nn','log',*CONTROLS)}
        score=lambda v:old.original.score(yall[qi],v,X,Xq,p79['splits'][s][0],p79['target_group_weights'])
        metrics={k:score(fixed[k]) for k in CONTROLS}
        assert metrics['v23']==a80['metrics']['R80'] and metrics['v22']==a['metrics']['R79']
        assert all(metrics[k]==a['metrics'][k] for k in BACKGROUND[1:])
        assert all(metrics[k]==a80['metrics'][k] for k in BACKGROUND)
        seed42_metrics=score(fixed['seed42']); assert seed42_metrics==a['unit_branch_metrics']['native']==a80['native_metrics']
        control_base=fixed['seed42']; exact(control_base,vector(paths['control_base']),'v23 base exact')
        p0,d0,rr0=old.original.predict_reference(reference,Xq,control_base,variant='trend',return_diagnostics=True)
        u0,detail0,sr0=spatial.predict(spatial_state,Xq,geo,control_base,p0,d0)
        exact(p0,vector(paths['control_reference']),'v23 reference exact'); exact(u0,vector(paths['control_spatial']),'v23 spatial exact')
        exact(np.median(np.column_stack([fixed['nn'],fixed['log'],u0]),axis=1),fixed['v23'],'v23 median exact')
        exact(u0[~detail0['applied'].to_numpy(dtype=bool)],p0[~detail0['applied'].to_numpy(dtype=bool)],'Control fallback')
        control_metrics={k:score(v) for k,v in (('base',control_base),('reference',p0),('spatial',u0))}
        assert control_metrics==a80['unit_branch_metrics']['candidate']
        assert all(sha(ROOT/name)==digest for name,digest in source.items())
        native, fit_seconds=fit_native(folder,s,X,yfit,G,Xq,Gq,cats,w,binding,frames,spec,result['plan_sha256'],inputs,result['runtime'])
        result['new_fits']+=1
        base=prices(.5*fixed['seed42']+.5*native,n)
        p,diag,rr=old.original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
        branch,detail,sr=spatial.predict(spatial_state,Xq,geo,base,p,diag)
        prices(p,n); prices(branch,n)
        fallback=~detail['applied'].to_numpy(dtype=bool); exact(branch[fallback],p[fallback],'Candidate fallback')
        candidate=prices(np.median(np.column_stack([fixed['nn'],fixed['log'],branch]),axis=1),n); assert (candidate>0).all()
        metrics['R83']=score(candidate)
        branch_metrics=dict(control=control_metrics,candidate={k:score(v) for k,v in (('base',base),('reference',p),('spatial',branch))})
        native_metrics=dict(seed42=seed42_metrics,seed137=score(native),mean=branch_metrics['candidate']['base'])
        for mode,B,P,U,D,S,V in (('control',control_base,p0,u0,d0,detail0,fixed['v23']),('candidate',base,p,branch,diag,detail,candidate)):
            for k,v in (('base',B),('reference',P),('unit_spatial',U),('predictions',V)): old.previous.save_array(folder/(mode+'_'+k+'.npy'),v)
            for k,v in (('reference_diagnostics',D),('spatial_diagnostics',S)):
                with (folder/(mode+'_'+k+'.pkl')).open('xb') as f: pickle.dump(v,f,protocol=5)
        old.previous.save_array(folder/'predictions.npy',candidate); exact(candidate,vector(rel(folder/'predictions.npy')),'Saved final exact')
        assert frames==dict(fit_X=spatial.frame_sha(X),query_X=spatial.frame_sha(Xq))
        assert spatial.frame_sha(G)==m19['fit_geo_sha256'] and spatial.frame_sha(Gq)==m19['query_geo_sha256']
        assert spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256']
        diff=old.previous.prior.log_run.differences
        item=dict(stage=s,status='completed',fit_rows=len(fi),query_rows=n,dates=p79['splits'][s],original_frame_sha256=frames,
            metrics=metrics,differences={k:diff(metrics[k],metrics['R83']) for k in CONTROLS},native_metrics=native_metrics,
            native_differences_vs_seed42={k:diff(seed42_metrics,native_metrics[k]) for k in ('seed137','mean')},
            unit_branch_metrics=branch_metrics,unit_branch_differences={k:diff(control_metrics[k],branch_metrics['candidate'][k]) for k in ('base','reference','spatial')},
            pipeline_reports=dict(control=dict(reference=rr0,spatial=sr0),candidate=dict(reference=rr,spatial=sr)),
            v23_base_reference_spatial_median_full_vectors_exact=True,controls_full_metrics_exact=True,
            original_R79_legacy_feature_state_exact=True,date_weight_provenance_verified=True,full_query_raw_TWD_reload_exact=True,
            inherited_seed42_reload_evidence_verified=True,seed42_retrained_or_model_reloaded=False,
            individual_seed_selection=False,query_labels_used_only_for_evaluation=True,spatial_fallback_exact=True,
            seed_prediction_paths={'42':paths['seed42'],'137':rel(folder/'model/native_predictions.npy')},
            changed_unit_rows=int(np.count_nonzero(branch!=u0)),changed_final_rows=int(np.count_nonzero(candidate!=fixed['v23'])),
            new_fits=1,new_NN_fits=0,reference_predict_calls=2,spatial_predict_calls=2,
            fit_seconds=fit_seconds,elapsed_seconds=time.monotonic()-start,prediction_sha256=sha(folder/'predictions.npy'),
            artifact_sha256={rel(p):sha(p) for p in folder.rglob('*') if p.is_file()})
        write(folder/'results.json',item); result['stages'][s]=item; result['checked_stages'].append(s)
        print(json.dumps(dict(stage=s,fit_seconds=fit_seconds,R83_vs_v23=item['differences']['v23']['overall'],new_fits=result['new_fits'])),flush=True)
    assert result['new_fits']==3 and result['checked_stages']==list(STAGES)
    assert all(sha(ROOT/name)==digest for name,digest in source.items()) and all(sha(ROOT/name)==digest for name,digest in inputs.items())
    after={rel(p):sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}; after['runs/versions.json']=sha(ROOT/'runs/versions.json')
    assert after==protected and sha(selector_path)==SELECTOR_SHA and sha(DOCUMENT)==PLAN_SHA
    result.update(status='three_periods_completed_no_adoption_decision',elapsed_seconds=time.monotonic()-tick,
        reference_predict_calls=6,spatial_predict_calls=6,protected_after=after,selector_sha256_after=sha(selector_path))
    write(OUT/'results.json',result)
    print(json.dumps(dict(status=result['status'],elapsed_seconds=result['elapsed_seconds'],result_sha256=sha(OUT/'results.json'),new_fits=3,new_CSV=False)),flush=True)


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='store_true',required=True); parser.add_argument('--plan-sha',required=True)
    args=parser.parse_args()
    assert args.plan_sha==PLAN_SHA==sha(DOCUMENT) and not OUT.exists(), 'Plan changed or prior attempt exists'
    OUT.mkdir(); tick,started,code,failure=time.monotonic(),old.now(),0,None
    write(OUT/'started.json',dict(command=sys.argv,started_at_utc=started,source_sha256=sha(Path(__file__)),automatic_retry=False))
    try: run(args.plan_sha)
    except Exception: code,failure=1,traceback.format_exc(); print(failure,flush=True)
    write(OUT/'execution.json',dict(command=sys.argv,exit_code=code,failure=failure,started_at_utc=started,
        completed_at_utc=old.now(),elapsed_seconds=time.monotonic()-tick,source_sha256=sha(Path(__file__)),automatic_retry=False))
    raise SystemExit(code)
