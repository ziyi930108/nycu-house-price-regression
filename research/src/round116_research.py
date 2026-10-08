"""R116: one fixed Depthwise unit CatBoost at four fixed origins, then scoring.

Source preparation is not fit authorization. The coordinator alone invokes
freeze/train/evaluate after independent preflight. All four native models,
raw unit outputs, reconstructed nonnegative TWD totals and complete five-layer chains are sealed and
training exits successfully before the separate scorer can read query prices.
No CSV, full-data fit, parameter search or shared split mutation is provided.
"""
from __future__ import annotations
import argparse
import contextlib
import json
import subprocess
import time
import traceback
from pathlib import Path
import numpy as np
import pandas as pd
import round100_contract as c
import round104_composition as comp
import round100_temporal_prior_evaluate_v2 as metric
import round111_112_research as history
import round113_offset_composition as offset_comp
import round113_offset_research as offset
import round97_offset_evaluate as offset_metric
import round116_depthwise_unit as helper

PLAN = 'reports/round116_plan.md'
PLAN_SHA = 'f9a4e0c8b7250d1fad5335d6fd02a92729d62bb8662d0376eeed286b8dd1cee6'
HELPER = 'src/round116_depthwise_unit.py'
BASE = 'outputs/round116'
EVIDENCE = 'outputs/research_round116_20261008'
REQUEST, SEAL, RESULTS = (BASE + '/' + name + '.json' for name in ('request', 'seal', 'results'))
OFFSET = 'offset_2023_02_11'
ORIGINS = (*c.STAGES, OFFSET)
LAYERS = ('native', 'base', 'reference', 'spatial', 'median')
COUNTS = {**{s: c.SPLITS[s][2:] for s in c.STAGES}, OFFSET: (272117, 107169)}
RAW_SHA = '9acf8c65bf00f7067e9ea9612cabe9a1524b020a7596c8da3ce3d620131e5c00'
CAUTION = ('Four reused overlapping origins are not independent votes or fresh blind validation. '
           'Saved v22/v24 controls are not matched refits. Conditional single-seed bootstrap '
           'omits cumulative selection, retraining variation and future shift; not Private probability. '
           'R27 deeper trees, R84 extra trees, R92 shallow unit and R112 Ordered supplied contrary evidence; '
           'native improvement alone cannot establish complete-composition improvement.')


def sources():
    return c.source_hashes(__file__, HELPER)


def protected():
    return history.protected()


def merge_hashes(target, incoming):
    for name, digest in incoming.items():
        c.need(name not in target or target[name] == digest, 'Conflicting input authority: ' + name)
        target[name] = digest


def trainer_recipe():
    """Constructor-only recipe; importing helper does not fit or read data."""
    old, new = helper.baseline.parameters(), helper.parameters()
    c.need(set(new) - set(old) == {'grow_policy'} and set(old) <= set(new)
           and all(new[k] == old[k] for k in old)
           and new['grow_policy'] == 'Depthwise' and new['boosting_type'] == 'Plain',
           'Only fixed grow_policy may be added to unchanged R79')
    return {'baseline_parameters': old, 'parameters': new, 'changed_parameter_names': [],
            'added_parameter_names': ['grow_policy'], 'baseline_implicit_grow_policy': 'SymmetricTree',
            'baseline_implicit_global_ctr_target_border_count': 1, 'implicit_min_data_in_leaf': 1,
            'native_derived_defaults_may_differ': True, 'target_mode': helper.TARGET_MODE,
            'target_formula': helper.TARGET_FORMULA, 'inverse_formula': helper.INVERSE_FORMULA,
            'sample_weight': helper.WEIGHT_FORMULA}


def formal_checks():
    return {'R116_helper_selfcheck_execution.json': (HELPER, '.venv', '--self-check'),
            'R116_driver_selfcheck_execution.json': (c.rel(__file__), '.venv', '--self-check')}


def check_formal_checks(inputs):
    for name, (entry, environment, flag) in formal_checks().items():
        path = EVIDENCE + '/' + name
        check = c.read(path)
        command = [str(c.ROOT / environment / 'Scripts/python.exe'), '-B', '-X', 'utf8', str(c.safe(entry)), flag]
        c.need(check['exit_code'] == 0 and check['command'] == command
               and check['source_sha256'] == c.source_hashes(entry), 'Formal source/command check differs: ' + name)
        inputs[path] = c.sha(path)


def august_specifications(inputs):
    names = ('outputs/round79/plan.json', 'outputs/round79/results.json', 'outputs/round63/results.json',
             'outputs/round83/results.json', 'outputs/round100_inputs/manifest.json')
    for name in names: inputs[name] = c.sha(name)
    p79, r79, r63, r83, labels = [c.read(name) for name in names]
    # Metadata/opaque identities only here. No raw price deserialization.
    merge_hashes(inputs, r63['input_sha256'])
    c.need(p79['input_sha256'][c.RAW] == RAW_SHA, 'Original scoring label identity')
    inputs[c.RAW] = RAW_SHA
    specs = {}
    for origin in c.STAGES:
        cs = comp.specification(origin); merge_hashes(inputs, cs['input_sha256'])
        old = r63['stages'][origin]
        prepared = f'outputs/round63/{origin}/prepared.npz'
        prep = f'outputs/round63/{origin}/preparation.json'
        training = f'outputs/round63/{origin}/model/training.json'
        for path in (prepared, prep, training): inputs[path] = old['artifact_sha256'][path]
        prior = c.read(prep)
        for key in ('prior_preparation', 'source_prepared'):
            path = prior[key + '_path']; digest = prior[key + '_sha256']
            merge_hashes(inputs, {path: digest})
        native = f'outputs/round79/{origin}/model/native_predictions.npy'
        inputs[native] = r79['stages'][origin]['artifact_sha256'][native]
        v24_names = {key: f'outputs/round83/{origin}/{name}' for key, name in (
            ('seed137', 'model/native_predictions.npy'), ('base', 'candidate_base.npy'),
            ('reference', 'candidate_reference.npy'), ('spatial', 'candidate_unit_spatial.npy'),
            ('median', 'predictions.npy'))}
        for path in v24_names.values(): inputs[path] = r83['stages'][origin]['artifact_sha256'][path]
        for record in labels['stages'][origin]['arrays'].values(): inputs[record['path']] = record['sha256']
        tree_geo = p79['bindings'][origin]['tree_geo']
        inputs[tree_geo] = p79['input_sha256'][tree_geo]
        tree_binding = p79['bindings'][origin]['R19_metadata']
        specs[origin] = {'tree_geo': c.record(tree_geo),
                        'tree_geo_frame_sha256': {k: tree_binding[k + '_geo_sha256'] for k in ('fit', 'query')},
                        'composition': cs, 'prepared': c.record(prepared), 'preparation': c.record(prep),
                        'original_training': c.record(training), 'native_unit': c.record(native),
                        'v24_layers': {k: c.record(v) for k, v in v24_names.items()},
                        'fit_labels': labels['stages'][origin]['arrays'],
                        'date_weight_sha256': p79['bindings'][origin]['preparation']['train_sample_weight_sha256']}
    return specs


def freeze():
    c.need(not c.safe(BASE).exists(), 'Preserve any previous R116 attempt')
    c.need(c.sha(PLAN) == PLAN_SHA, 'Fixed pre-scoring four-origin plan changed')
    own = sources(); review_path = EVIDENCE + '/preflight_review.json'; review = c.read(review_path)
    c.need(review['status'] == 'passed' and review['source_sha256'] == {**own, PLAN: PLAN_SHA},
           'Independent complete source/plan preflight required')
    c.verify_hashes(review['source_sha256'])
    inputs = {review_path: c.sha(review_path)}; check_formal_checks(inputs)
    recipe = trainer_recipe(); inputs['reports/feature_schema.json'] = c.sha('reports/feature_schema.json')
    specs = august_specifications(inputs)
    extra = offset_comp.specification(); merge_hashes(inputs, extra['input_sha256'])
    inputs['outputs/round97/evaluation/results.json'] = c.sha('outputs/round97/evaluation/results.json')
    old_seal = c.read(extra['original_seal']['path'])
    c.need(old_seal['input_sha256'][c.RAW] == inputs[c.RAW] == RAW_SHA, 'All scoring-only raw axes agree')
    specs[OFFSET] = {'composition': extra, 'prepared': extra['prior_npz']}
    c.verify_hashes(inputs)
    c.safe(BASE).mkdir(exist_ok=False)
    c.write(REQUEST, {'status': 'frozen_before_four_fixed_origin_fits', 'created_at_utc': c.now(),
        'round': 116, 'candidate': 'R116', 'origin_order': list(ORIGINS), 'origin_counts': COUNTS,
        'plan': c.record(PLAN), 'origins': specs, 'source_sha256': own, 'input_sha256': inputs,
        'recipe': recipe, 'protected_before': protected(), 'new_fits': 4,
        'query_scores_computed': False, 'query_labels_used_for_fit': False, 'public_tuning': False,
        'Public_score': None, 'Private_score': None, 'selected_submission': 'v22',
        'reserved_exploratory_version': 31, 'new_CSV': False, 'full_data_fit': False})


def request():
    r = c.read(REQUEST)
    c.need(r['status'] == 'frozen_before_four_fixed_origin_fits' and r['round'] == 116
           and r['candidate']=='R116' and r['origin_order'] == list(ORIGINS) and set(r['origins']) == set(ORIGINS)
           and r['origin_counts']=={key:list(value) for key,value in COUNTS.items()}
           and r['new_fits'] == 4 and r['new_CSV'] is False and r['full_data_fit'] is False,
           'Immutable four-origin request scope differs')
    c.need(r['plan'] == c.record(PLAN) and r['plan']['sha256'] == PLAN_SHA, 'Plan changed')
    c.need(r['source_sha256'] == sources() and r['recipe'] == trainer_recipe(), 'Frozen source/recipe changed')
    c.verify_hashes(r['source_sha256']); c.verify_hashes(r['input_sha256'])
    c.need(protected() == r['protected_before'], 'Protected original artifacts changed')
    return r


def prepared_arrays(path, nf, nq, fit_y):
    """Check all archive names first, then materialize fit date weights/y only.

    Unit fitting does not need NN encoded features, price factors or query labels.
    Offset context independently retains its original nine-array replay contract.
    """
    with np.load(c.safe(path), allow_pickle=False) as archive:
        c.need(len(archive.files) == len(c.ARRAY_NAMES) and set(archive.files) == set(c.ARRAY_NAMES),
               'Exactly nine prepared keys before materializing; no query y')
        arrays = {key: archive[key].copy() for key in ('train_sample_weight', 'train_y_million')}
    w, y = arrays['train_sample_weight'], arrays['train_y_million']
    c.need(w.dtype == np.float32 and w.shape == (nf,) and np.isfinite(w).all() and (w > 0).all(),
           'Exact positive saved float32 fit-date weights')
    c.need(y.dtype == np.float32 and y.shape == (nf,) and np.isfinite(y).all() and (y >= 0).all(),
           'Exact prepared fit-target dtype/shape/nonnegative values')
    c.exact(y, np.asarray(fit_y / 1e6, dtype=np.float32), 'Fit-only canonical y alignment')
    return arrays


def replay_august_v24(ctx, spec):
    """R83 mean of native42/native137, with its original R/S; no LGB added."""
    native42 = np.load(c.safe(spec['native_unit']['path']), allow_pickle=False)
    seed137 = np.load(c.safe(spec['v24_layers']['seed137']['path']), allow_pickle=False)
    control = history.unit_chain(ctx, native42)
    c.exact(control['spatial'], ctx['fixed']['U'], 'Original v22 unit spatial replay')
    c.exact(control['median'], ctx['original']['median'], 'Original v22 full median replay')
    base = .5 * native42 + .5 * seed137
    ref, diag, _ = comp.old.original.predict_reference(ctx['reference'], ctx['query_X'], base, variant='trend', return_diagnostics=True)
    spatial, detail, _ = comp.old.spatial.predict(ctx['spatial_state'], ctx['query_X'], ctx['geo'], base, ref, diag)
    fallback = ~detail['applied'].to_numpy(dtype=bool)
    c.exact(spatial[fallback], ref[fallback], 'Original v24 spatial fallback')
    median = np.median(np.column_stack([ctx['original']['spatial'], ctx['fixed']['H'], spatial]), axis=1)
    for key, value in {'base': base, 'reference': ref, 'spatial': spatial, 'median': median}.items():
        c.exact(value, np.load(c.safe(spec['v24_layers'][key]['path']), allow_pickle=False), 'Original v24 ' + key + ' replay')
    comp.verify_unchanged(ctx)
    return median


def context_identity(ctx, origin):
    if origin == OFFSET:
        return {'state': ctx['state_identity'], 'frames_and_indices': ctx['data_identity'],
                'fit_labels': ctx['fit_label_identity'], 'weights': ctx['weight_identity'],
                'baseline_vectors': ctx['vector_identity']}
    return {'state': ctx['state_identity'], 'frames': ctx['frame_identity'],
            'indices': {k: c.identity(v) for k, v in ctx['indices'].items()},
            'fixed': {k: c.identity(v) for k, v in ctx['fixed'].items()},
            'original': {k: c.identity(v) for k, v in ctx['original'].items()}}


def load_context(r, origin):
    c.need(origin in ORIGINS, 'Unknown origin')
    spec = r['origins'][origin]; nf, nq = COUNTS[origin]
    if origin == OFFSET:
        ctx = offset_comp.load_context(spec['composition'])
        y, v24 = ctx['yfit'], ctx['baseline']['v24']
        X, Xq, G, Gq = ctx['data'].Xfit, ctx['data'].Xquery, ctx['data'].tree_fit, ctx['data'].tree_query
        indices = {'fit_indices': ctx['data'].fit_indices,
                   'fit_source_cache_indices': ctx['data'].fit_source_cache_indices,
                   'query_indices': ctx['data'].query_indices,
                   'query_source_cache_indices': ctx['data'].query_source_cache_indices}
    else:
        # Actual frozen API: comp.load_context; no nonexistent context/feature_arrays.
        ctx = comp.load_context(spec['composition']); labels = spec['fit_labels']
        for key in ('fit_indices', 'fit_source_cache_indices'):
            c.exact(c.load_array(labels[key]), ctx['indices'][key], 'Fit label axis: ' + key)
        y = c.load_array(labels['fit_y']); v24 = replay_august_v24(ctx, spec)
        X, Xq, indices = ctx['fit_X'], ctx['query_X'], ctx['indices']
        c.need(c.record(spec['tree_geo']['path']) == spec['tree_geo'], 'Frozen tree geography changed')
        tree = pd.read_pickle(c.safe(spec['tree_geo']['path']))
        G = tree.iloc[indices['fit_indices']].reset_index(drop=True)
        Gq = tree.iloc[indices['query_indices']].reset_index(drop=True); del tree
        c.need({key: helper.base.frame_sha(value) for key,value in (('fit',G),('query',Gq))}
               == spec['tree_geo_frame_sha256'], 'Original-cutoff fit/query geography axes')
    c.need(len(X) == nf and len(Xq) == nq and y.dtype == np.float64 and y.shape == (nf,)
           and np.isfinite(y).all() and (y > 0).all(), 'Fixed origin fit labels/row counts')
    arrays = prepared_arrays(spec['prepared']['path'], nf, nq, y)
    identities = {key: c.identity(value) for key, value in arrays.items()}
    if origin == OFFSET:
        c.need(identities == {key:ctx['prepared_array_identities'][key] for key in arrays}, 'Offset weight/y identity')
        c.exact(arrays['train_sample_weight'], ctx['weights'], 'Offset saved dates')
    else:
        old = c.read(spec['original_training']['path'])
        c.need(identities['train_sample_weight']['values_sha256'] == old['train_sample_weight_sha256']
               == spec['date_weight_sha256'], 'Original date weight authority')
    cats = c.read('reports/feature_schema.json')['categorical_columns']
    c.need(len(cats) == len(set(cats)) == 21 and set(cats) <= set(X.columns), '21 original input categories')
    for frame in (helper.original.augment_features(X,G), helper.original.augment_features(Xq,Gq)):
        c.need(len(frame.columns) == 95 and len(helper.original.categories(X,cats)) == 23,
               'Unchanged 95 augmented features / 23 categories')
    result = {'context':ctx, 'fit_y':y, 'arrays':arrays, 'array_identities':identities,
              'fit_X':X, 'query_X':Xq, 'fit_geo':G, 'query_geo':Gq, 'cats':cats,
              'indices':indices, 'v24':v24, 'v22':ctx['baseline']['v22'] if origin==OFFSET else ctx['original']['median'],
              'before':context_identity(ctx,origin)}
    result['fit_input_identity'] = fit_input_identity(result)
    return result


def fit_input_identity(loaded):
    X,Xq,G,Gq = (loaded[k] for k in ('fit_X','query_X','fit_geo','query_geo'))
    y = loaded['fit_y']; w = loaded['arrays']['train_sample_weight']
    frames = {'fit_X':helper.base.frame_sha(X), 'query_X':helper.base.frame_sha(Xq),
              'fit_geo':helper.base.frame_sha(G), 'query_geo':helper.base.frame_sha(Gq),
              'fit_augmented':helper.base.frame_sha(helper.original.augment_features(X,G)),
              'query_augmented':helper.base.frame_sha(helper.original.augment_features(Xq,Gq))}
    return {'frame_sha256':frames, 'fit_target_values_sha256':helper.original.values_sha256(y),
            'fit_transformed_target_values_sha256':helper.original.values_sha256(helper.original.transform_target(y,X)),
            'fit_date_weights_sha256':helper._date_sha(w),
            'fit_area_weights_sha256':helper.original.values_sha256(helper.original.sample_weights(X)),
            'fit_sample_weights_sha256':helper.original.values_sha256(helper.sample_weights(X,w)),
            'categorical_columns':list(loaded['cats'])}


def verify_context(loaded, origin):
    ctx = loaded['context']
    (offset_comp.verify_unchanged if origin == OFFSET else comp.verify_unchanged)(ctx)
    c.need(context_identity(ctx, origin) == loaded['before'], 'Fit/query state or original vectors mutated')
    c.need({k:c.identity(v) for k,v in loaded['arrays'].items()} == loaded['array_identities'], 'Prepared inputs mutated')
    c.need(fit_input_identity(loaded) == loaded['fit_input_identity'], 'Unit fit/query/target/geography/weights mutated')


def unit_inverse_diagnostics(raw, native, Xq):
    rows = len(Xq)
    c.need(rows > 0 and raw.dtype == native.dtype == np.float64 and raw.shape == native.shape == (rows,)
           and np.isfinite(raw).all() and np.isfinite(native).all(), 'Finite float64 raw unit/native total')
    c.exact(helper.original.inverse_transform(raw,Xq),native,'Exactly one original unit-to-TWD inverse')
    return {'raw_units':helper.TARGET_MODE, 'raw_negative_unit_rows':int((raw<0).sum()),
            'raw_unit_min':float(raw.min()), 'raw_unit_max':float(raw.max()),
            'native_TWD_min':float(native.min()), 'native_TWD_max':float(native.max()),
            'native_zero_rows':int((native==0).sum()), 'inverse_applications_per_prediction':1,
            'nonnegative_projection_after_TWD_reconstruction':True,
            'query_y_used_for_diagnostics':False, 'extra_price_cap_applied':False}


def check_metadata(meta,r,origin,loaded):
    nf,nq=COUNTS[origin]; recipe=r['recipe']; identity=loaded['fit_input_identity']
    c.need(meta['status']=='completed' and meta['component']==helper.COMPONENT
           and meta['parameters']==recipe['parameters'] and meta['baseline_parameters']==recipe['baseline_parameters']
           and meta['changed_parameter_names']==[] and meta['added_parameter_names']==['grow_policy']
           and meta['source_sha256']==helper.source_hashes()
           and all(r['source_sha256'][key]==digest for key,digest in meta['source_sha256'].items())
           and meta['fit_rows']==nf and meta['query_rows']==nq and meta['new_native_fits']==1
           and meta['feature_count']==95 and meta['input_categorical_feature_count']==21
           and meta['categorical_feature_count']==23 and meta['target_mode']==recipe['target_mode']
           and meta['target_formula']==recipe['target_formula'] and meta['inverse_formula']==recipe['inverse_formula']
           and meta['sample_weight']==recipe['sample_weight'] and meta['weight_dtype']=='float64'
           and meta['weight_argument_passed_to_pool'] is True and meta['date_weights_recomputed'] is False
           and meta['date_weights_renormalized'] is False and meta['weights_normalized_to_mean_one'] is False
           and meta['weights_use_fit_target'] is False and meta['weights_use_query'] is False
           and meta['date_weight_provenance_verified_in_helper'] is False
           and meta['legacy_state_parameters_are_provenance_not_training'] is True
           and meta['legacy_state_training_fields_are_provenance_only'] is True
           and meta['query_labels_available'] is False and meta['query_labels_used'] is False
           and meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False
           and meta['saved_model_reload_exact'] is True and meta['raw_and_TWD_reload_exact'] is True
           and meta['reload_rows']==nq and meta['inverse_applications_per_prediction']==1
           and meta['reference_or_trend_applied_here'] is False and meta['spatial_applied_here'] is False
           and meta['extra_price_cap_applied'] is False and meta['new_CSV'] is False
           and meta['automatic_fallback_used'] is False and meta['parameters_retuned_after_failure'] is False,
           'Actual fixed Depthwise unit metadata differs')
    for key,value in identity.items():
        if key!='categorical_columns':c.need(meta[key]==value,'Actual fit input identity: '+key)
    c.need(meta['actual_grow_policy']=='Depthwise' and meta['actual_boosting_type']=='Plain'
           and meta['actual_task_type']=='GPU' and meta['actual_model_depth']==8 and meta['actual_trees']==2999
           and meta['actual_seed']==42 and meta['actual_max_ctr_complexity']==2
           and meta['actual_ctr_target_border_count']==1 and meta['actual_min_data_in_leaf']==1
           and meta['tree_leaf_counts_count']==2999 and 1<=meta['minimum_tree_leaf_count']<=meta['maximum_tree_leaf_count']<=256
           and sum(meta['tree_leaf_counts_distribution'].values())==2999
           and meta['runtime_parameters_not_recovered_from_native_model']=={
               k:recipe['parameters'][k] for k in helper.NATIVE_RUNTIME_ONLY_PARAMETER_NAMES}, 'Native policy/capacity/runtime declaration')


def model_records(folder):
    files=sorted(folder.iterdir())
    required={'model.cbm','started.json','metadata.json','feature_state.json','raw.npy','native.npy'}
    c.need(files and all(p.is_file() and p.suffix in {'.json','.npy','.cbm'} for p in files)
           and {p.name for p in files}==required,'Bind every flat native CatBoost artifact, no NN files')
    return [c.record(path) for path in files]


def validate(origin,record,r,loaded=None):
    folder=c.safe(BASE+'/'+origin); model=folder/'model'; nf,nq=COUNTS[origin]
    c.need(record==c.record(folder/'stage.json'),'Stage exact identity')
    item=c.read(record['path'])
    c.need(item['stage']==origin and item['candidate_round']==116 and item['fit_rows']==nf and item['query_rows']==nq
           and item['query_labels_available'] is False and item['query_scores_computed'] is False
           and item['training_log']==c.record(folder/'train.log')
           and item['model_metadata']==c.record(model/'metadata.json')
           and item['sealed_models']==model_records(model) and item['context_before']==item['context_after'],
           'Stage scope/native artifact/state seal differs')
    loaded=load_context(r,origin) if loaded is None else loaded
    c.need(item['fit_input_identity']==loaded['fit_input_identity']
           and item['prepared_array_identities']==loaded['array_identities'],'Actual frozen unit input identities')
    meta=c.read(model/'metadata.json'); check_metadata(meta,r,origin,loaded)
    start=c.read(model/'started.json')
    c.need(all(meta.get(k)==v for k,v in start.items()),'Native before/after metadata contract')
    for file,key in (('model.cbm','model_sha256'),('feature_state.json','feature_state_sha256'),
                     ('raw.npy','raw_prediction_sha256'),('native.npy','prediction_sha256')):
        c.need(c.sha(model/file)==meta[key],'Direct native artifact SHA: '+file)
    state=c.read(model/'feature_state.json')
    helper._check_state(state,loaded['fit_X'],loaded['cats']); helper._check_state(state,loaded['query_X'],loaded['cats'])
    restored=helper.CatBoostRegressor(); restored.load_model(str(model/'model.cbm'))
    actual=helper.check_model(restored,state)
    c.need(actual==meta['reloaded_native_diagnostics'],'Actual CPU native CBM capacity/default/leaf identity')
    helper.compare_reload_diagnostics(meta['fit_native_diagnostics'],actual)
    raw,native=[np.load(model/name,allow_pickle=False) for name in ('raw.npy','native.npy')]
    diagnostic=unit_inverse_diagnostics(raw,native,loaded['query_X'])
    c.need(item['unit_inverse_diagnostics']==diagnostic
           and item['raw_unit_prediction']=={**c.record(model/'raw.npy'),**c.identity(raw)},'Raw unit inverse seal')
    layers={k:c.load_array(v) for k,v in item['layers'].items()}
    c.need(set(layers)==set(LAYERS),'Exactly five unit composition layers')
    for key,values in layers.items():metric.prices(values,(nq,),key+' complete layer')
    c.exact(layers['native'],native,'Native file/five-layer alignment')
    verify_context(loaded,origin)
    return item,layers,raw


def stage(origin):
    r=request(); started=c.read(EVIDENCE+'/training_started.json')
    c.need(started['request']==c.record(REQUEST) and started['new_fits_authorized']==4,'Coordinated serial training marker required')
    loaded=load_context(r,origin); nf,nq=COUNTS[origin]
    folder=c.safe(BASE+'/'+origin); folder.mkdir(exist_ok=False)
    with (folder/'train.log').open('x',encoding='utf8') as log:
        with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
            native,meta=helper.fit_predict(loaded['fit_X'],loaded['fit_y'],loaded['fit_geo'],
                loaded['query_X'],loaded['query_geo'],loaded['cats'],loaded['arrays']['train_sample_weight'],folder/'model')
    check_metadata(meta,r,origin,loaded)
    raw=np.load(folder/'model/raw.npy',allow_pickle=False)
    diagnostic=unit_inverse_diagnostics(raw,native,loaded['query_X'])
    ctx=loaded['context']; chain=offset_comp.chain_unit(ctx,native) if origin==OFFSET else history.unit_chain(ctx,native)
    c.need(set(chain)==set(LAYERS),'Exactly one unit42 slot replacement and complete composition')
    records={key:c.save_array(folder/(key+'.npy'),value) for key,value in chain.items()}
    verify_context(loaded,origin)
    c.write(folder/'stage.json',{'stage':origin,'candidate_round':116,'fit_rows':nf,'query_rows':nq,
        'model_metadata':c.record(folder/'model/metadata.json'),'sealed_models':model_records(folder/'model'),
        'training_log':c.record(folder/'train.log'),
        'raw_unit_prediction':{**c.record(folder/'model/raw.npy'),**c.identity(raw)},
        'layers':records,'prepared_array_identities':loaded['array_identities'],'fit_input_identity':loaded['fit_input_identity'],
        'fit_seconds':meta['fit_seconds'],'context_before':loaded['before'],'context_after':context_identity(ctx,origin),
        'unit_inverse_diagnostics':diagnostic,'original_v22_v24_and_unit42_slot_replayed_exact':True,
        'query_labels_available':False,'query_scores_computed':False})
    validate(origin,c.record(folder/'stage.json'),r,loaded); request()
    print(json.dumps({'round':116,'origin':origin,'status':'saved_and_sealed_unscored'}),flush=True)


def train():
    r = request(); tick = time.monotonic(); failure = None
    c.need(not c.safe(EVIDENCE + '/training_started.json').exists()
           and not c.safe(SEAL).exists() and not c.safe(EVIDENCE + '/training_execution.json').exists(), 'One serial attempt only')
    c.write(EVIDENCE + '/training_started.json', {'request': c.record(REQUEST), 'created_at_utc': c.now(),
            'new_fits_authorized': 4, 'automatic_retry': False})
    try:
        for origin in ORIGINS:
            command = [str(c.ROOT / '.venv/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(Path(__file__).resolve()), '--stage', origin]
            process = subprocess.run(command, cwd=c.ROOT, check=False)
            c.need(process.returncode == 0, 'Origin failed, no automatic retry: ' + origin)
        records = {s: c.record(BASE + '/' + s + '/stage.json') for s in ORIGINS}
        for origin, record in records.items(): validate(origin, record, r)
        request()
        c.write(SEAL, {'status': 'four_models_all_raw_unit_TWD_and_layers_sealed_unscored',
            'request': c.record(REQUEST), 'origin_order': list(ORIGINS), 'records': records, 'new_fits': 4,
            'query_scores_computed': False, 'elapsed_seconds': time.monotonic() - tick})
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(EVIDENCE + '/training_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
            'request': c.record(REQUEST), 'elapsed_seconds': time.monotonic() - tick, 'automatic_retry': False})


def combined_gates(stages):
    c.need(set(stages) == set(ORIGINS), 'All four fixed origins required')
    august = history.gates({s: stages[s] for s in c.STAGES})
    extra = offset.gates(stages[OFFSET]['metrics'], stages[OFFSET]['comparison'])
    result = {'august': august, 'extra_origin': extra}
    for name, old in (('combined_exploratory_submission', august['exploratory_submission']['passed']),
                      ('combined_promotion', august['promotion']['adoption_research_gate_passed'])):
        parts = {'original_three_windows': old, 'fixed_extra_origin': extra['passed']}
        result[name] = {'components': parts, 'passed': all(parts.values())}
    result.update(private_score=None, selected_submission='v22', automatic_selection=False,
                  new_CSV=False, full_data_fit=False, reserved_exploratory_version=31)
    return result


def scoring_preflight():
    r = request(); seal = c.read(SEAL); execution = c.read(EVIDENCE + '/training_execution.json')
    c.need(seal['status'] == 'four_models_all_raw_unit_TWD_and_layers_sealed_unscored'
           and seal['request'] == c.record(REQUEST) and seal['origin_order'] == list(ORIGINS)
           and set(seal['records']) == set(ORIGINS) and seal['new_fits'] == 4 and seal['query_scores_computed'] is False,
           'All four models/layers must be sealed before any query price access')
    c.need(execution['exit_code'] == 0 and execution['failure'] is None and execution['request'] == c.record(REQUEST),
           'Serial training must exit successfully before separate scoring')
    checked = {s: validate(s, seal['records'][s], r) for s in ORIGINS}
    c.need(r['input_sha256'][c.RAW] == RAW_SHA and c.sha(c.RAW) == RAW_SHA, 'Scoring-only raw label source changed')
    return r, seal, checked


def evaluate():
    c.need(not c.safe(RESULTS).exists() and not c.safe(EVIDENCE + '/evaluation_execution.json').exists(), 'Preserve any previous scoring attempt')
    r, seal, checked = scoring_preflight(); tick = time.monotonic(); failure = None
    try:
        # First and only raw-price deserialization: after every model/metadata/layer and successful exit guard.
        raw = pd.read_pickle(c.safe(c.RAW)); stages = {}
        for origin in ORIGINS:
            loaded = load_context(r, origin); item, layers, raw_unit = checked[origin]
            if origin == OFFSET:
                descriptive = lambda v: offset_metric.descriptive_metrics(y, v, loaded['fit_X'], loaded['query_X'])
                y = offset_metric.select_query_y(raw, loaded['indices']['query_source_cache_indices'], 455784)
            else:
                y = metric.select_query_y(raw, loaded['indices']['query_source_cache_indices'])
                descriptive = lambda v: metric.descriptive_metrics(y, v, loaded['fit_X'], loaded['query_X'], c.SPLITS[origin][0])
            metrics = {'v22': descriptive(loaded['v22']), 'v24': descriptive(loaded['v24']), 'T': descriptive(layers['median'])}
            if origin == OFFSET:
                old = c.read('outputs/round97/evaluation/results.json')['metrics']
                c.need(all(metrics[k] == old[k] for k in ('v22', 'v24')), 'Original offset control metrics replay')
                comparison = offset_metric.comparisons({'v22': metrics['v22'], 'v24': metrics['T']})
                comparison24 = offset_metric.comparisons({'v22': metrics['v24'], 'v24': metrics['T']})
                paired, samples = offset_metric.paired(y, {'v22': loaded['v22'], 'v24': layers['median']},
                    loaded['fit_X'], loaded['query_X'], loaded['indices']['query_indices'])
            else:
                for key, number, label in (('v22', 79, 'R79'), ('v24', 83, 'R83')):
                    old = c.read(f'outputs/round{number}/results.json')['stages'][origin]['metrics'][label]
                    c.need(all(metrics[key][k] == old[k] for k in old), 'Original August control metrics replay: ' + key)
                comparison, comparison24 = metric.comparisons(metrics), metric.comparisons(metrics, 'v24')
                paired, samples = metric.paired(y, {'v22': loaded['v22'], 'T': layers['median']},
                    loaded['fit_X'], loaded['query_X'], loaded['indices']['query_indices'], origin, 'v22')
            comparison.update(control='v22', candidate='R116', used_for_release_gate=True)
            comparison24.update(control='v24', candidate='R116', used_for_release_gate=False)
            paired.update(candidate='R116', difference='R116 minus v22', caution=CAUTION)
            folder = c.safe(BASE + '/' + origin + '/evaluation'); folder.mkdir(exist_ok=False)
            sample_records = {k: c.save_array(folder / (k + '_samples.npy'), v) for k, v in samples.items()}
            result = {'metrics': metrics, 'comparison': comparison, 'comparison_vs_v24': comparison24,
                'paired': paired, 'paired_samples': sample_records, 'sealed_stage': seal['records'][origin],
                'branch_metrics': {k: descriptive(v) for k, v in layers.items() if k != 'median'},
                'raw_unit_diagnostic_only': {**item['unit_inverse_diagnostics'],
                    'unprojected_TWD_overall_mae': float(np.abs(raw_unit * helper.original._price_factor(loaded['query_X']) - y).mean())},
                'both_point_metrics_improve': all(metrics['T'][k] < metrics['v22'][k] for k in ('overall_mae', 'weighted_mae'))}
            verify_context(loaded, origin); c.write(folder / 'results.json', result); stages[origin] = result
            print(json.dumps({'round': 116, 'origin': origin, 'mae': metrics['T']['overall_mae'],
                  'delta': metrics['T']['overall_mae'] - metrics['v22']['overall_mae']}), flush=True)
        del raw
        scoring_preflight()
        c.write(RESULTS, {'status': 'four_reused_fixed_origins_evaluated', 'created_at_utc': c.now(),
            'round': 116, 'request': c.record(REQUEST), 'seal': c.record(SEAL), 'plan': c.record(PLAN),
            'source_sha256': sources(), 'origin_order': list(ORIGINS), 'stages': stages, 'gates': combined_gates(stages),
            'new_fits': 4, 'new_CSV': False, 'full_data_fit': False, 'Public_score': None, 'Private_score': None,
            'selected_submission': 'v22', 'fresh_blind_validation': False, 'caution': CAUTION})
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(EVIDENCE + '/evaluation_execution.json', {'exit_code': 1 if failure else 0, 'failure': failure,
            'elapsed_seconds': time.monotonic() - tick, 'request': c.record(REQUEST), 'seal': c.record(SEAL),
            'training_execution': c.record(EVIDENCE + '/training_execution.json'),
            'query_labels_used_for_scoring_only': True, 'training_already_finished_before_scoring': True,
            'new_fits': 0, 'new_CSV': False})


def self_check():
    """Synthetic input/gate/score guards only; no real data/model load or fit."""
    from unittest.mock import patch
    checks=[]
    def ok(name,condition):c.need(condition,name);checks.append(name)
    def rejects(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError('Guard accepted: '+name)
    source_before=sources(); split_before=dict(c.SPLITS)
    Xq=pd.DataFrame({'building_area':np.array([2.,0.,-1.],np.float32),
                     'parking_area':np.array([9.,3.,0.],np.float32)})
    raw=np.array([-1.,0.,2.],np.float64)
    native=np.array([0.,0.,20000.],np.float64)
    info=unit_inverse_diagnostics(raw,native,Xq)
    ok('raw_unit_inverse_original_area_and_one_projection',info['raw_negative_unit_rows']==1 and info['native_zero_rows']==2)
    rejects('raw_unit_not_signed_total_or_double_inverse',lambda:unit_inverse_diagnostics(raw,native+1,Xq))
    rejects('unit_nan_rejected',lambda:unit_inverse_diagnostics(np.array([np.nan,0.,2.]),native,Xq))
    rejects('unit_float32_rejected',lambda:unit_inverse_diagnostics(raw.astype(np.float32),native,Xq))
    rejects('empty_unit_rejected',lambda:unit_inverse_diagnostics(np.empty(0),np.empty(0),Xq.iloc[:0]))
    ok('four_origins_without_global_SPLITS_mutation',ORIGINS==('development','confirmation','auxiliary',OFFSET)
       and COUNTS=={'development':(240316,81234),'confirmation':(321550,133697),'auxiliary':(134805,105511),OFFSET:(272117,107169)}
       and dict(c.SPLITS)==split_before)
    recipe=trainer_recipe(); old=recipe['baseline_parameters']; new=recipe['parameters']
    ok('only_added_Depthwise_keeps_Plain_original_recipe',set(new)-set(old)=={'grow_policy'}
       and all(new[k]==old[k] for k in old) and new['boosting_type']=='Plain'
       and recipe['changed_parameter_names']==[] and recipe['added_parameter_names']==['grow_policy'])
    ok('CPU_helper_and_driver_formal_checks_are_explicit',formal_checks()=={
       'R116_helper_selfcheck_execution.json':(HELPER,'.venv','--self-check'),
       'R116_driver_selfcheck_execution.json':(c.rel(__file__),'.venv','--self-check')})
    def sample(a,e,strong):
        oldgate={'exploratory_submission':{'passed':a},'promotion':{'adoption_research_gate_passed':strong}}
        stages={s:{'metrics':{},'comparison':{}} for s in ORIGINS}
        with patch.object(history,'gates',return_value=oldgate),patch.object(offset,'gates',return_value={'passed':e}):
            return combined_gates(stages)
    for a in (False,True):
        for e in (False,True):
            for strong in (False,True):
                result=sample(a,e,strong)
                ok('combined_gate_'+str(int(a))+str(int(e))+str(int(strong)),
                   result['combined_exploratory_submission']['passed']==(a and e)
                   and result['combined_promotion']['passed']==(strong and e)
                   and result['reserved_exploratory_version']==31 and result['new_CSV'] is False)
    rejects('missing_offset_gate_rejected',lambda:combined_gates({s:{} for s in c.STAGES}))
    class FakeArchive:
        def __init__(self,arrays,files=None):self.arrays,self.files,self.reads=arrays,list(c.ARRAY_NAMES) if files is None else files,[]
        def __enter__(self):return self
        def __exit__(self,*args):return False
        def __getitem__(self,key):self.reads.append(key);return self.arrays[key]
    arrays={'train_sample_weight':np.array([.25,.5,1.],np.float32),'train_y_million':np.ones(3,np.float32)}
    truth=np.full(3,1e6,np.float64);archive=FakeArchive(arrays)
    with patch.object(np,'load',return_value=archive):
        actual=prepared_arrays('synthetic.npz',3,3,truth)
    ok('all_nine_key_check_but_only_fit_weight_and_y_materialized',set(actual)==set(arrays)
       and archive.reads==['train_sample_weight','train_y_million'])
    for i,files in enumerate((list(c.ARRAY_NAMES)+['query_y'],list(c.ARRAY_NAMES[:-1])+['query_y'],list(c.ARRAY_NAMES)+['train_num'])):
        archive=FakeArchive(arrays,files)
        with patch.object(np,'load',return_value=archive):
            rejects('invalid_archive_keys_before_values_'+str(i),lambda:prepared_arrays('synthetic.npz',3,3,truth))
        ok('invalid_archive_zero_reads_'+str(i),not archive.reads)
    for name,key,bad in [('zero_weight','train_sample_weight',np.zeros(3,np.float32)),
                         ('nan_weight','train_sample_weight',np.full(3,np.nan,np.float32)),
                         ('wrong_weight_dtype','train_sample_weight',np.ones(3,np.float64)),
                         ('wrong_weight_length','train_sample_weight',np.ones(2,np.float32)),
                         ('wrong_y_dtype','train_y_million',np.ones(3,np.float64)),
                         ('unaligned_fit_y','train_y_million',np.zeros(3,np.float32))]:
        archive=FakeArchive({**arrays,key:bad})
        with patch.object(np,'load',return_value=archive):
            rejects(name,lambda:prepared_arrays('synthetic.npz',3,3,truth))
    # A real raw-price load must remain behind complete source/model and exit guards.
    dummy={'request':{},'status':'bad'}
    with patch.dict(globals(),{'request':lambda:{}}),patch.object(c,'read',return_value=dummy),patch.object(c,'record',return_value={}):
        rejects('incomplete_seal_refuses_scoring',scoring_preflight)
    sealed={'status':'four_models_all_raw_unit_TWD_and_layers_sealed_unscored','request':{},
            'origin_order':list(ORIGINS),'records':{s:{} for s in ORIGINS},'new_fits':4,'query_scores_computed':False}
    execution={'exit_code':0,'failure':None,'request':{}}
    with patch.dict(globals(),{'request':lambda:{'input_sha256':{c.RAW:RAW_SHA}},'validate':lambda *args:({}, {},raw)}), \
         patch.object(c,'read',side_effect=lambda name:sealed if name==SEAL else execution), \
         patch.object(c,'record',return_value={}),patch.object(c,'sha',return_value=RAW_SHA):
        ok('four_native_validations_complete_before_raw_y_access',len(scoring_preflight()[2])==4)
        execution['exit_code']=1
        rejects('failed_training_exit_refuses_scoring',scoring_preflight);execution['exit_code']=0
        sealed['records'].pop(OFFSET)
        rejects('missing_fourth_native_seal_refuses_scoring',scoring_preflight)
    # Check successful receipt command/source binding and reject a wrong command,
    # closure or exit status without running a subprocess or reading real receipt.
    expected={}
    for name,(entry,environment,flag) in formal_checks().items():
        expected[EVIDENCE+'/'+name]={'exit_code':0,'command':[str(c.ROOT/environment/'Scripts/python.exe'),'-B','-X','utf8',str(c.safe(entry)),flag],
                                   'source_sha256':c.source_hashes(entry)}
    original_sha=c.sha
    with patch.object(c,'read',side_effect=lambda name:expected[name]), \
         patch.object(c,'sha',side_effect=lambda name:'synthetic' if name in expected else original_sha(name)):
        bound={};check_formal_checks(bound)
        ok('all_formal_CPU_receipts_mandatory',set(bound)==set(expected))
        key=next(iter(expected));saved=expected[key]
        for label,bad in [('exit',{'exit_code':1}),('command',{'command':['wrong']}),('closure',{'source_sha256':{}})]:
            expected[key]={**saved,**bad}
            rejects('formal_'+label+'_rejects',lambda:check_formal_checks({}))
        expected[key]=saved
    ok('complete_source_and_global_SPLITS_unchanged',sources()==source_before and dict(c.SPLITS)==split_before)
    ok('unique_CPU_check_names',len(checks)==len(set(checks)))
    return {'status':'passed','checks':checks,'check_count':len(checks),'source_sha256':c.source_hashes(__file__),
            'complete_research_source_sha256':sources(),'query_labels_read':False,'real_data_read':False,
            'new_fits':0,'GPU_calls':0,'new_CSV':False,'global_SPLITS_modified':False,
            'independent_preflight_required_before_freeze_and_fit':True}


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__); group=parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze',action='store_true'); group.add_argument('--train',action='store_true')
    group.add_argument('--stage',choices=ORIGINS); group.add_argument('--evaluate',action='store_true')
    group.add_argument('--self-check',action='store_true'); args=parser.parse_args()
    if args.freeze: freeze()
    elif args.train: train()
    elif args.stage: stage(args.stage)
    elif args.evaluate: evaluate()
    else: print(json.dumps(self_check(),ensure_ascii=False,sort_keys=True,allow_nan=False),flush=True)
