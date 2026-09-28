"""R102: saved R100T inputs, one log-prior column versus same-width positive zero.

Six serial R63 NN fits; stage children never read raw/query prices or fit_y files.
No prior, encoder or R/S state fitting. Real work requires a frozen root request.
"""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import pickle
import subprocess
import sys
import time
import traceback
import numpy as np
import pandas as pd
import round100_contract as c
import round48_nn_experiment as old
import round19_unit_geo as area_helper

PLAN = 'reports/round102_plan.md'
PLAN_SHA = 'ee59dba0da8abb79c88cb6e1b73a6417a91944de784f3c7f77e61755a68bc355'
OUT = c.ROOT / 'outputs/round102'
REQUEST = 'outputs/round102_launcher_20260929/research_request.json'
R100_SEAL = 'outputs/round100/sealed_manifest.json'
AUTHORITY_PATHS = (R100_SEAL, 'outputs/round100/execution.json',
    'outputs/round100_independent_review/results.json', 'outputs/round101/results.json',
    'outputs/round101/execution.json', 'outputs/round48/plan.json', 'outputs/round60/results.json')
ELIGIBLE = {'development': 240260, 'confirmation': 321494, 'auxiliary': 134749}
FORMULA = 'median(S(R(.75*(.5*R54_area+.5*prior_NN)+.25*unit_LGB)),R46_H,R79_U)'


def source_hashes():
    return c.source_hashes(__file__, c.TRAINER)


def parameters():
    return {'round': 102, 'arm_order': ['C', 'T'], 'arm_roles': {'C': 'positive_zero_control', 'T': 'log_prior_candidate'},
            'numeric_count': 145, 'column_position_zero_based': 144, 'source': 'same-stage saved R100T nine arrays',
            'log_scaling': 'float64 natural log; eligible fit mean/std ddof=0; float32 column',
            'std_must_be_positive': True, 'clipping': False, 'date_weights_renormalized': False,
            'seed': 42, 'epochs': 30, 'batch_size': 2048, 'prior_build_calls': 0,
            'stage_order': list(c.STAGES), 'planned_new_NN_fits': 6, 'new_CSV': False}


def check_launch(path):
    request = c.read(path)
    c.need(request['status'] == 'frozen_before_real_prior_or_fit' and request['approval_for_six_fits'] is True,
           'Frozen six-fit request required')
    c.need(c.sha(PLAN) == request['plan_sha256'] == PLAN_SHA, 'R102 plan differs')
    c.need(request['source_sha256'] == source_hashes(), 'Source closure differs')
    c.need(all(request['frozen_source_and_plan'].get(p) == h for p, h in
               {**request['source_sha256'], PLAN: PLAN_SHA}.items()), 'Incomplete source freeze')
    c.need(set(AUTHORITY_PATHS) <= set(request['input_authority_sha256']), 'Missing input authority')
    c.need(all(Path(p).suffix == '.json' for p in request['input_authority_sha256']), 'Metadata-only authority gate')
    c.verify_hashes(request['frozen_source_and_plan']); c.verify_hashes(request['input_authority_sha256'])
    parent = c.read(R100_SEAL)
    c.need(parent['plan_sha256'] == c.PLAN_SHA == c.sha(c.PLAN), 'Original R100 plan identity differs')
    c.verify_hashes(parent['source_sha256'])
    c.need(parent['protected_before'] == parent['protected_after'] == request['protected_before'], 'R100 protection changed')
    c.need(c.sha(c.TRAINER) == c.TRAINER_SHA and c.protected_hashes() == request['protected_before'], 'Trainer/protection differs')
    return request


def positive(a, rows, dtype, name, zero=False):
    c.need(isinstance(a, np.ndarray) and a.dtype == np.dtype(dtype) and a.shape == (rows,)
           and np.isfinite(a).all() and ((a >= 0).all() if zero else (a > 0).all()), 'Invalid ' + name)


def augment_arrays(original, train_prior, query_prior):
    """Pure array transform. Both arms use the already-eligible R100T factors."""
    c.need(set(original) == set(c.ARRAY_NAMES), 'Exactly nine source arrays required')
    nf, nq = len(original['train_y_million']), len(original['query_num'])
    positive(train_prior, nf, 'float64', 'temporal train unit prior')
    positive(query_prior, nq, 'float64', 'full-fit query unit prior')
    for split, rows in [('train', nf), ('query', nq)]:
        a = original[split+'_num']
        c.need(a.dtype == np.float32 and a.shape == (rows, 144) and np.isfinite(a).all(), 'Source numeric schema')
    before = {k: c.identity(a) for k, a in original.items()}
    train_log, query_log = np.log(train_prior), np.log(query_prior)
    mean, std = float(train_log.mean(dtype=np.float64)), float(train_log.std(ddof=0, dtype=np.float64))
    c.need(np.isfinite(mean) and np.isfinite(std) and std > 0, 'Log prior std must be positive finite; no fallback')
    features = {'train': ((train_log-mean)/std).astype(np.float32),
                'query': ((query_log-mean)/std).astype(np.float32)}
    c.need(all(np.isfinite(a).all() for a in features.values()), 'Nonfinite added feature')
    arms = {arm: {k: a.copy() for k, a in original.items()} for arm in ('C', 'T')}
    for split, rows in [('train', nf), ('query', nq)]:
        key = split+'_num'
        arms['C'][key] = np.column_stack([original[key], np.zeros(rows, dtype=np.float32)])
        arms['T'][key] = np.column_stack([original[key], features[split]])
        for arm in arms:
            c.exact(arms[arm][key][:, :144], original[key], 'Original numeric prefix differs')
        c.exact(arms['C'][key][:, 144], np.zeros(rows, dtype=np.float32), 'Control is not positive zero')
    for key in set(c.ARRAY_NAMES)-{'train_num', 'query_num'}:
        for arm in arms: c.exact(arms[arm][key], original[key], 'Shared source array changed: '+key)
    c.need(before == {k: c.identity(a) for k, a in original.items()}, 'Caller source mutated')
    scale = {'format': 'R102_log_temporal_unit_prior_v1', 'feature': 'standardized_log_temporal_unit_prior',
             'numeric_column_position_zero_based': 144, 'fit_rows': nf, 'query_rows': nq,
             'mean_log_float64': mean, 'std_log_ddof0_float64': std,
             'train_prior_identity': c.identity(train_prior), 'query_prior_identity': c.identity(query_prior),
             'train_feature_identity': c.identity(features['train']), 'query_feature_identity': c.identity(features['query']),
             'query_scaling_uses_fit_only': True, 'clipping': False}
    return arms, scale


def save_npz(path, arrays):
    c.need(set(arrays) == set(c.ARRAY_NAMES), 'Exactly nine trainer arrays')
    with path.open('xb') as f: np.savez(f, **arrays)
    with np.load(path, allow_pickle=False) as z:
        c.need(set(z.files) == set(c.ARRAY_NAMES), 'Saved NPZ key mismatch')
        for key in c.ARRAY_NAMES: c.exact(arrays[key], z[key], 'Saved NPZ differs: '+key)
    return {k: c.identity(a) for k, a in arrays.items()}


def train_arm(folder, arrays, baseline):
    folder.mkdir(); prepared = folder/'prepared.npz'; identities = save_npz(prepared, arrays)
    cmd = [str(old.TORCH_PYTHON), '-u', '-B', '-X', 'utf8', str(c.safe(c.TRAINER)), '--data', str(prepared), '--out', str(folder/'model')]
    c.write(folder/'fit_started.json', {'started_at_utc': c.now(), 'command': cmd, 'prepared': c.record(prepared),
                                      'all_input_arrays': identities, 'automatic_retry': False})
    tick = time.monotonic()
    with (folder/'trainer.log').open('x', encoding='utf8') as log:
        done = subprocess.run(cmd, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    c.write(folder/'execution.json', {'exit_code': done.returncode, 'failure': None if done.returncode == 0 else 'trainer subprocess failed',
                                     'elapsed_seconds': time.monotonic()-tick, 'command': cmd, 'automatic_retry': False})
    c.need(done.returncode == 0, 'Trainer failed; no automatic retry')
    m = c.read(folder/'model/training.json'); nf, nq = len(arrays['train_y_million']), len(arrays['query_num'])
    c.need(m['status'] == 'completed' and m['configuration'] == baseline['configuration'], 'R63 configuration differs')
    c.need(m['source_sha256'] == c.TRAINER_SHA and m['architecture_source_sha256'] == old.ARCHITECTURE_SHA, 'Trainer source differs')
    c.need(m['input_sha256'] == c.sha(prepared) and (m['fit_rows'], m['query_rows'], m['numeric_count']) == (nf, nq, 145), 'Trainer input differs')
    c.need(m['cardinalities'] == arrays['cardinalities'].tolist() and m['saved_model_reload_exact'] and m['reload_rows'] == nq, 'Model schema/reload differs')
    c.need(m['query_labels_available'] is False and m['query_labels_used'] is False and not m['sample_weights_batch_renormalized']
           and not m['price_or_ratio_clipping'], 'Trainer changed label/weight contract')
    c.need([e['epoch'] for e in m['epochs']] == list(range(1,31)), 'Incomplete 30 epochs')
    for key in ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
        c.need(m[key+'_sha256'] == identities[key]['values_sha256'] == baseline[key+'_sha256'], 'Factor/date weights differ')
    for key in ('initial_training_median_ratio', 'initial_head_bias'):
        c.need(m[key] == baseline[key], 'Original temporal target bias differs')
    native = np.load(folder/'model/predictions_twd.npy', allow_pickle=False); positive(native, nq, 'float64', 'native prediction')
    return native, {'prepared': c.record(prepared), 'training': c.record(folder/'model/training.json'),
        'model': c.record(folder/'model/model.pt'), 'execution': c.record(folder/'execution.json'),
        'all_input_arrays': identities, 'fit_seconds': m['fit_seconds'], 'saved_model_reload_exact': True, 'reload_rows': nq,
        'original_R63_configuration_exact': True, 'native_prediction': {**c.record(folder/'model/predictions_twd.npy'), **c.identity(native)}}


def stage_run(stage, request_path):
    request = check_launch(request_path)
    c.need(OUT.is_dir() and c.read(OUT/'started.json')['request'] == c.record(request_path), 'Stage lacks serial parent')
    folder = OUT/stage; c.need(not folder.exists(), 'Stage exists; no overwrite'); folder.mkdir()
    tick = time.monotonic(); code = 1; failure = None; inputs = {}; before = c.protected_hashes()
    try:
        def checked(name, digest):
            name = c.rel(name)
            c.need(name != c.RAW and not name.startswith(('data_cache/test', 'data_cache/valid'))
                   and not name.endswith('/fit_y.npy'), 'Forbidden price input')
            if name.endswith('.npz'): c.need(name == f'outputs/round100/{stage}/T/prepared.npz', 'Only own-stage R100T NPZ allowed')
            if name.startswith('outputs/round100/') and name.endswith(('.npy','.npz','.pkl')):
                c.need(name.startswith(f'outputs/round100/{stage}/'), 'Other-stage R100 data forbidden')
            c.need(name not in inputs or inputs[name] == digest, 'Conflicting input identity')
            if name not in inputs: c.need(c.sha(name) == digest, 'Input changed: '+name); inputs[name] = digest
            return c.safe(name)
        def doc(rec): return c.read(checked(rec['path'], rec['sha256']))
        def arr(rec): checked(rec['path'], rec['sha256']); return c.load_array(rec)
        authority = {p: c.read(checked(p, request['input_authority_sha256'][p])) for p in AUTHORITY_PATHS}
        parent = authority[R100_SEAL]
        c.need(parent['status'] == 'all_temporal_prior_predictions_sealed' and parent['plan_sha256'] == c.PLAN_SHA
               and parent['new_NN_fits'] == 6 and parent['all_query_reload_exact'], 'R100 source not completed')
        for p in ('outputs/round100/execution.json', 'outputs/round101/execution.json'):
            c.need(authority[p]['exit_code'] == 0 and authority[p]['failure'] is None, 'Prerequisite failed')
        old_stage_rec = parent['stages'][stage]['stage_manifest']; source = doc(old_stage_rec)
        ex = doc(parent['stages'][stage]['execution'])
        c.need(ex['exit_code'] == 0 and ex['failure'] is None and source['stage'] == stage
               and source['plan_sha256'] == c.PLAN_SHA and source['all_query_reload_exact'], 'Incomplete original stage')
        start, end, nf, nq = c.SPLITS[stage]; ne = ELIGIBLE[stage]
        c.need((source['cutoff'], source['query_end'], source['fit_rows'], source['query_rows'], source['eligible_rows'])
               == (start, end, nf, nq, ne), 'Original split differs')
        ids = {k: arr(source[k]) for k in ('fit_indices','query_indices','fit_source_cache_indices','query_source_cache_indices','eligible_positions')}
        fi, qi, kept = ids['fit_indices'], ids['query_indices'], ids['eligible_positions']
        for a, n in ((fi,nf),(qi,nq),(kept,ne)):
            c.need(a.dtype == np.int64 and a.shape == (n,) and (np.diff(a)>0).all() and (a>=0).all(), 'Indices schema/order')
        c.need(kept[-1] < nf and not np.intersect1d(fi,qi).size, 'Eligible/outer overlap')
        mapping = np.load(checked(source['canonical_mapping']['path'], source['canonical_mapping']['sha256']), allow_pickle=False)
        c.need(mapping.dtype == np.int64 and mapping.shape == (455255,) and (np.diff(mapping)>0).all(), 'Canonical mapping schema')
        c.exact(mapping[fi], ids['fit_source_cache_indices'], 'Raw fit mapping'); c.exact(mapping[qi], ids['query_source_cache_indices'], 'Raw query mapping')
        Xraw = pd.read_pickle(checked(source['features']['path'], source['features']['sha256']))
        c.need(len(Xraw) == 455784 and '總價元' not in Xraw and Xraw.index.equals(pd.RangeIndex(len(Xraw))), 'Feature cache schema')
        X = Xraw.iloc[mapping[fi]].reset_index(drop=True); Xq = Xraw.iloc[mapping[qi]].reset_index(drop=True); del Xraw
        day = lambda d: (pd.Timestamp(d)-pd.Timestamp('2000-01-01')).days
        c.need(np.isfinite(X.trade_day_index).all() and (X.trade_day_index < day(start)).all()
               and Xq.trade_day_index.between(day(start),day(end)).all(), 'Outer time contract differs')
        prepared_rec = source['arms']['T']['prepared']; prepared = checked(prepared_rec['path'], prepared_rec['sha256'])
        with np.load(prepared, allow_pickle=False) as z:
            c.need(set(z.files) == set(c.ARRAY_NAMES), 'Exactly nine source keys required before materializing')
            original = {k:z[k].copy() for k in c.ARRAY_NAMES}
        c.need({k:c.identity(a) for k,a in original.items()} == source['arms']['T']['all_input_arrays'], 'Saved R100T array identity differs')
        baseline = doc(source['arms']['T']['training'])
        c.need(baseline['source_sha256'] == c.TRAINER_SHA and baseline['input_sha256'] == c.sha(prepared)
               and (baseline['fit_rows'],baseline['query_rows'],baseline['numeric_count']) == (ne,nq,144), 'Source trainer binding differs')
        prior_train = arr(source['temporal_prior'])
        query_name = f'outputs/round60/{stage}/query_unit_prior.npy'
        query_digest = authority['outputs/round60/results.json']['stages'][stage]['artifact_sha256'][query_name]
        c.need(source['original_input_sha256'][query_name] == query_digest, 'Query prior authority mismatch')
        prior_query = np.load(checked(query_name,query_digest),allow_pickle=False)
        positive(prior_train,ne,'float64','temporal prior'); positive(prior_query,nq,'float64','query prior')
        for key,A,P in [('train_price_factor_million',area_helper.effective_area(X)[kept],prior_train),
                        ('query_price_factor_million',area_helper.effective_area(Xq),prior_query)]:
            c.exact(np.asarray(A*P/1e6,dtype=np.float32),original[key],'Saved temporal/fullfit factor mismatch')
        arms_arrays, scale = augment_arrays(original,prior_train,prior_query)
        c.write(folder/'scale_state.json',scale)
        preparation = {'status':'matched_prior_input_arms_prepared','source_prepared_R100T':prepared_rec,
            'source_input_arrays':source['arms']['T']['all_input_arrays'], 'original_144_prefix_identity':{k:c.identity(original[k]) for k in ('train_num','query_num')},
            'only_arm_numeric_column_difference':144,'other_seven_arrays_bitwise_unchanged':True,'date_weights_renormalized':False,
            'arm_feature_identity':{arm:{split:c.identity(arms_arrays[arm][split+'_num'][:,144]) for split in ('train','query')} for arm in ('C','T')},
            'scale_state':c.record(folder/'scale_state.json'),'prior_records':{'train':source['temporal_prior'],
              'query':{**c.record(query_name),**c.identity(prior_query)}},'query_labels_available':False,'prior_build_calls':0}
        c.write(folder/'preparation.json',preparation)
        p48 = authority['outputs/round48/plan.json']; b = p48['bindings'][stage]
        def bound(name):
            c.need(source['original_input_sha256'][name] == p48['input_sha256'][name], 'R/S source authority differs')
            return checked(name,p48['input_sha256'][name])
        with bound(b['reference']).open('rb') as f: reference = pickle.load(f)
        with bound(b['state']).open('rb') as f: spatial_state = pickle.load(f)
        summary = c.read(bound(b['state_summary'])); geo = pd.read_pickle(bound(b['query_geo']))
        c.need(spatial_state['state_summary'] == summary and summary['fit_X_sha256'] == old.spatial.frame_sha(X)
               and not summary['query_used_in_build'] and reference['fit_report']['fitting_rows'] == nf, 'R/S fullfit scope differs')
        c.need(spatial_state['parameters'] == old.spatial.parameters() and geo.index.equals(Xq.index)
               and old.spatial.frame_sha(geo) == b['manifest']['features']['query']['frame_sha256'], 'R/S geography differs')
        state_before = hashlib.sha256(pickle.dumps((reference,spatial_state),protocol=5)).hexdigest()
        frames_before = [old.spatial.frame_sha(x) for x in (X,Xq)]
        fixed = {k:arr(rec) for k,rec in source['fixed_branches'].items()}
        c.need(set(fixed)=={'area','lgb','H','U'}, 'Fixed branches differ')
        predictions = {k:arr(source['predictions'][k]) for k in ('v22','v24')}; layers = {}; arms = {}
        def chain(native,name):
            positive(native,nq,'float64','native',zero=True)
            base = .75*(.5*fixed['area']+.5*native)+.25*fixed['lgb']
            ref,diag,_ = old.original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
            spatial,detail,_ = old.spatial.predict(spatial_state,Xq,geo,base,ref,diag)
            mask = ~detail['applied'].to_numpy(dtype=bool); c.exact(spatial[mask],ref[mask],'Spatial fallback')
            values = {'native':native,'base':base,'reference':ref,'spatial':spatial}
            if name == 'v22':
                for k,a in values.items(): c.exact(a,arr(source['layers']['v22'][k]),'Original v22 layer '+k)
                for key,frame in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
                    path=f'outputs/round100/{stage}/v22/{key}.pkl'
                    with checked(path,source['artifact_sha256'][path]).open('rb') as f: pd.testing.assert_frame_equal(frame,pickle.load(f),check_exact=True)
            saved=folder/name; saved.mkdir(exist_ok=True)
            layers[name]={k:c.save_array(saved/(k+'.npy'),a) for k,a in values.items()}
            for key,frame in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
                with (saved/(key+'.pkl')).open('xb') as f: pickle.dump(frame,f,protocol=5)
                with (saved/(key+'.pkl')).open('rb') as f: pd.testing.assert_frame_equal(frame,pickle.load(f),check_exact=True)
            return np.median(np.column_stack([spatial,fixed['H'],fixed['U']]),axis=1)
        c.exact(chain(arr(source['layers']['v22']['native']),'v22'),predictions['v22'],'Original v22 complete median')
        for arm in ('C','T'):
            c.verify_hashes(request['source_sha256']); c.verify_hashes(inputs)
            native,arms[arm]=train_arm(folder/arm,arms_arrays[arm],baseline)
            predictions[arm]=chain(native,arm)
            print(json.dumps({'stage':stage,'arm':arm,'fit_seconds':arms[arm]['fit_seconds'],'full_query_reload_exact':True}),flush=True)
        c.need({k:c.identity(a) for k,a in original.items()}==source['arms']['T']['all_input_arrays'], 'Source NPZ changed in memory')
        c.need(frames_before==[old.spatial.frame_sha(x) for x in (X,Xq)] and state_before==hashlib.sha256(pickle.dumps((reference,spatial_state),protocol=5)).hexdigest(), 'Original state mutated')
        c.verify_hashes(inputs); c.verify_hashes(request['source_sha256']); c.need(before==c.protected_hashes(),'Protected files changed')
        seal={'schema_version':1,'status':'prior_input_stage_predictions_sealed','stage':stage,'cutoff':start,'query_end':end,
            'fit_rows':nf,'query_rows':nq,'eligible_rows':ne,'numeric_count':145,'new_NN_fits':2,'new_native_tree_fits':0,
            'prior_build_calls':0,'query_scores_computed':False,'query_labels_available':False,'raw_price_pickle_opened':False,
            'fit_y_separate_file_opened':False,'other_stage_fit_labels_opened':False,'all_query_reload_exact':True,
            'original_v22_full_chain_and_diagnostics_exact':True,'original_R63_configuration_exact':True,'same_factors_weights_and_bias':True,
            'plan_sha256':PLAN_SHA,'parameters':parameters(),'request':c.record(request_path),'source_sha256':request['source_sha256'],
            'source_R100_plan_sha256':c.PLAN_SHA,'source_R100_seal':c.record(R100_SEAL),'source_R100_stage_seal':old_stage_rec,
            'source_prepared_R100T':prepared_rec,'source_input_arrays':source['arms']['T']['all_input_arrays'],
            'features':source['features'],'canonical_mapping':source['canonical_mapping'],'raw_train':source['raw_train'],
            'original_input_sha256':inputs,'protected_before':before,'protected_after':c.protected_hashes(),
            'arms':arms,'layers':layers,'formula':FORMULA,'scale_state':c.record(folder/'scale_state.json'),
            'preparation':c.record(folder/'preparation.json'),'fit_seconds':sum(a['fit_seconds'] for a in arms.values()),
            'elapsed_seconds':time.monotonic()-tick}
        for key,a in ids.items(): seal[key]=c.save_array(folder/(key+'.npy'),a)
        seal['fixed_branches']={k:c.save_array(folder/(k+'.npy'),a) for k,a in fixed.items()}
        seal['predictions']={k:c.save_array(folder/('prediction_'+k+'.npy'),a) for k,a in predictions.items()}
        seal['artifact_sha256']={c.rel(p):c.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        c.write(folder/'stage_seal.json',seal); code=0
    except Exception:
        failure=traceback.format_exc(); raise
    finally:
        c.write(folder/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
                                       'stage':stage,'query_scores_computed':False,'automatic_retry':False})


def pipeline(request_path):
    request=check_launch(request_path); c.need(not OUT.exists(),'R102 output exists; no overwrite'); OUT.mkdir()
    tick=time.monotonic(); code=1; failure=None
    c.write(OUT/'started.json',{'started_at_utc':c.now(),'request':c.record(request_path),'plan_sha256':PLAN_SHA,'automatic_retry':False})
    try:
        stages={}
        for stage in c.STAGES:
            cmd=[sys.executable,'-u','-B','-X','utf8',str(Path(__file__).resolve()),'--stage',stage,'--request',str(c.safe(request_path))]
            with (OUT/(stage+'_process.log')).open('x',encoding='utf8') as f: done=subprocess.run(cmd,cwd=c.ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
            c.need(done.returncode==0,'Stage failed; no automatic retry: '+stage)
            folder=OUT/stage; st=c.read(folder/'stage_seal.json'); ex=c.read(folder/'execution.json')
            c.need(st['new_NN_fits']==2 and st['all_query_reload_exact'] and ex['exit_code']==0 and ex['failure'] is None,'Incomplete stage')
            stages[stage]={'stage_manifest':c.record(folder/'stage_seal.json'),'execution':c.record(folder/'execution.json'),
                           'predictions':st['predictions'],'fit_seconds':st['fit_seconds']}
        c.verify_hashes(request['frozen_source_and_plan']); c.verify_hashes(request['input_authority_sha256'])
        c.need(request['protected_before']==c.protected_hashes(),'Protected files changed')
        parent=c.read(R100_SEAL)
        seal={'schema_version':1,'status':'all_prior_input_predictions_sealed','stage_order':list(c.STAGES),
            'new_NN_fits':6,'new_native_tree_fits':0,'prior_build_calls':0,'query_scores_computed':False,'query_labels_available':False,
            'raw_price_pickle_opened':False,'fit_y_separate_file_opened':False,'other_stage_fit_labels_opened':False,'all_query_reload_exact':True,
            'plan_sha256':PLAN_SHA,'parameters':parameters(),'source_sha256':request['source_sha256'],'request':c.record(request_path),
            'source_R100_plan_sha256':c.PLAN_SHA,'source_R100_seal':c.record(R100_SEAL),
            'features':parent['features'],'canonical_mapping':parent['canonical_mapping'],'raw_train':parent['raw_train'],
            'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),'stages':stages,
            'fit_seconds':sum(s['fit_seconds'] for s in stages.values()),'elapsed_seconds':time.monotonic()-tick,
            'new_CSV':False,'selected_submission':'v22','Private_score':None}
        c.write(OUT/'sealed_manifest.json',seal); code=0
        print(json.dumps({'status':seal['status'],'new_NN_fits':6,'elapsed_seconds':seal['elapsed_seconds']}),flush=True)
    except Exception:
        failure=traceback.format_exc(); raise
    finally:
        c.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
            'source_sha256':c.sha(__file__),'automatic_retry':False,'query_scores_computed':False})


def self_check():
    tick=time.monotonic(); checks=[]
    def mark(name,ok): c.need(bool(ok),name); checks.append(name)
    x=np.arange(4*144,dtype=np.float32).reshape(4,144)
    a={'train_num':x,'train_cat':np.zeros((4,2),dtype=np.int64),'train_y_million':np.arange(1,5,dtype=np.float32),
       'query_num':x[:2].copy(),'query_cat':np.zeros((2,2),dtype=np.int64),'cardinalities':np.array([1,1],dtype=np.int64),
       'train_price_factor_million':np.array([.5,1.,2.,4.],dtype=np.float32),'query_price_factor_million':np.array([2.,3.],dtype=np.float32),
       'train_sample_weight':np.array([.1,.2,1.,3.4],dtype=np.float32)}
    p=np.array([10000.,20000.,40000.,80000.]);q=np.array([15000.,1e8]);before={k:c.identity(v) for k,v in a.items()}
    arms,scale=augment_arrays(a,p,q); expected=(np.log(p)-np.log(p).mean())/np.log(p).std(ddof=0)
    mark('float64_log_then_float32',np.array_equal(arms['T']['train_num'][:,-1],expected.astype(np.float32)))
    mark('fit_only_mean_std',scale['mean_log_float64']==float(np.log(p).mean()) and scale['std_log_ddof0_float64']==float(np.log(p).std(ddof=0)))
    mark('query_not_clipped',arms['T']['query_num'][1,-1]>arms['T']['train_num'][:,-1].max())
    for key in c.ARRAY_NAMES:
        if key not in ('train_num','query_num'): mark('unchanged_'+key,c.identity(arms['C'][key])==c.identity(arms['T'][key])==before[key])
    for split in ('train','query'):
        key=split+'_num'; mark(split+'_prefix',all(c.identity(arms[arm][key][:,:144])==before[key] for arm in arms))
        zero=arms['C'][key][:,144]; mark(split+'_positive_zero',(zero==0).all() and not np.signbit(zero).any())
        mark(split+'_width145',arms['C'][key].shape[1]==arms['T'][key].shape[1]==145)
    changed,newscale=augment_arrays(a,p,q*3)
    mark('query_does_not_change_fit',c.identity(changed['T']['train_num'])==c.identity(arms['T']['train_num']) and newscale['mean_log_float64']==scale['mean_log_float64'] and newscale['std_log_ddof0_float64']==scale['std_log_ddof0_float64'])
    swapped,ss=augment_arrays(a,p,q[::-1].copy()); mark('query_order_preserved',np.array_equal(swapped['T']['query_num'][:,-1],arms['T']['query_num'][::-1,-1]))
    mark('source_not_mutated',before=={k:c.identity(v) for k,v in a.items()})
    mark('scale_JSON_roundtrip',json.loads(json.dumps(scale,allow_nan=False))==scale)
    for label,tp,qp in [('zero',p*0,q),('negative',-p,q),('nonfinite',p,q*np.inf),('wrong_length',p[:-1],q),('float32_prior',p.astype(np.float32),q),('zero_std',np.ones(4,dtype=np.float64),q)]:
        try: augment_arrays(a,tp,qp)
        except ValueError: checks.append('reject_'+label)
        else: raise AssertionError('Accepted invalid '+label)
    bad={**a,'query_labels':np.ones(2)}
    try: augment_arrays(bad,p,q)
    except ValueError: checks.append('reject_extra_label_key')
    else: raise AssertionError('Accepted extra key')
    return {'status':'passed','exit_code':0,'check_count':len(checks),'checks':checks,'source_sha256':c.sha(__file__),
            'plan_sha256':PLAN_SHA,'parameters':parameters(),'real_data_read':False,'new_real_native_fits':0,
            'prior_build_calls':0,'query_scores_computed':False,'GPU_calls':0,'trainer_execution_tested':False,
            'elapsed_seconds':time.monotonic()-tick}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true'); mode.add_argument('--stage',choices=c.STAGES); mode.add_argument('--self-check',action='store_true')
    p.add_argument('--request',default=REQUEST); p.add_argument('--receipt'); args=p.parse_args()
    if args.self_check:
        c.need(args.receipt is not None,'Synthetic receipt path required'); target=c.safe(args.receipt)
        c.need(not target.exists() and not target.with_suffix('.sha256').exists(),'Receipt exists')
        try: result=self_check()
        except Exception:
            c.write(target,{'status':'failed','exit_code':1,'failure':traceback.format_exc(),'source_sha256':c.sha(__file__),
                            'real_data_read':False,'new_real_native_fits':0}); raise
        c.write(target,result); print(json.dumps(result),flush=True)
    elif args.stage: stage_run(args.stage,args.request)
    else: pipeline(args.request)
