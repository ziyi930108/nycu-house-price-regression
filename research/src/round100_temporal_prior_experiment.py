"""R100 six serial NN fits in three isolated stage processes; never query scoring.

Each child opens only its own fit-label NPY and original R63 NPZ. The parent
loads metadata only. The separate evaluator is launched externally after seal.
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
import round100_temporal_prior as helper
import round48_nn_experiment as old
import round47_nn_encode as encoder
import round6_geo as geo_encoder
import round19_unit_geo as area_helper
import round60_recency_price_prior as price_prior

OUT = c.ROOT / 'outputs/round100'
REQUEST = 'outputs/round100_launcher_20260929/research_request.json'
HISTORY = ('outputs/round48/plan.json', 'outputs/round79/plan.json') + tuple(
    f'outputs/round{r}/results.json' for r in (48, 54, 60, 63, 71, 79, 83))
AUTHORITY_PATHS = HISTORY + (c.X_AUDIT, c.FIT_LABELS)
FORMULA = 'median(S(R(.75*(.5*R54_area+.5*prior_NN)+.25*unit_LGB)),R46_H,R79_U)'


def source_hashes():
    return c.source_hashes(__file__, c.TRAINER)


def check_launch(path):
    request = c.read(path)
    c.need(request['status'] == 'frozen_before_real_prior_or_fit' and request['approval_for_six_fits'] is True,
           'Six-fit research request required')
    c.need(c.sha(c.PLAN) == c.PLAN_SHA == request['plan_sha256'], 'Frozen plan differs')
    c.need(request['source_sha256'] == source_hashes(), 'Driver/trainer source closure differs')
    required = {**request['source_sha256'], c.PLAN: c.PLAN_SHA}
    c.need(all(request['frozen_source_and_plan'].get(p) == h for p, h in required.items()), 'Incomplete source freeze')
    c.verify_hashes(request['frozen_source_and_plan'])
    c.need(set(AUTHORITY_PATHS) <= set(request['input_authority_sha256']), 'Missing input authority')
    # Only metadata is opened here; stage arrays/labels have separate read sets.
    c.need(all(Path(p).suffix == '.json' for p in request['input_authority_sha256']), 'Launch input gate must be JSON-only')
    c.verify_hashes(request['input_authority_sha256'])
    c.need(c.sha(c.TRAINER) == c.TRAINER_SHA and request['protected_before'] == c.protected_hashes(), 'Trainer/protected bytes changed')
    return request


def positive(a, rows, dtype, name, zero=False):
    c.need(a.dtype == np.dtype(dtype) and a.shape == (rows,) and np.isfinite(a).all()
           and ((a >= 0).all() if zero else (a > 0).all()), 'Invalid ' + name)


def masked_arrays(original, mask, temporal_prior, area):
    """The two arms differ only in train factor; no factor on excluded rows."""
    c.need(mask.dtype == bool and mask.shape == original['train_y_million'].shape and mask.any(), 'Invalid eligible mask')
    rows = int(mask.sum())
    positive(temporal_prior, rows, 'float64', 'temporal unit prior')
    common = {k: (a[mask].copy() if k.startswith('train_') else a.copy()) for k, a in original.items()}
    C = {k: a.copy() for k, a in common.items()}
    T = {k: a.copy() for k, a in common.items()}
    T['train_price_factor_million'] = np.asarray(area[mask] * temporal_prior / 1e6, dtype=np.float32)
    for k in c.ARRAY_NAMES:
        if k != 'train_price_factor_million':
            c.exact(C[k], T[k], 'Arm changed a shared array: ' + k)
    for k in ('train_y_million', 'train_sample_weight', 'train_price_factor_million'):
        positive(C[k], rows, 'float32', 'C ' + k)
        positive(T[k], rows, 'float32', 'T ' + k)
    c.exact(C['train_sample_weight'], original['train_sample_weight'][mask], 'Loss weights renormalized')
    return {'C': C, 'T': T}


def save_npz(path, arrays):
    c.need(set(arrays) == set(c.ARRAY_NAMES), 'Exactly nine trainer arrays required')
    with path.open('xb') as f:
        np.savez(f, **arrays)
    with np.load(path, allow_pickle=False) as z:
        c.need(set(z.files) == set(c.ARRAY_NAMES), 'Saved NPZ keys differ')
        for k in c.ARRAY_NAMES:
            c.exact(arrays[k], z[k], 'Saved NPZ array differs: ' + k)
    return {k: c.identity(a) for k, a in arrays.items()}


def train_arm(folder, arrays, baseline):
    folder.mkdir()
    prepared = folder / 'prepared.npz'
    identities = save_npz(prepared, arrays)
    command = [str(old.TORCH_PYTHON), '-u', '-B', '-X', 'utf8', str(c.safe(c.TRAINER)),
               '--data', str(prepared), '--out', str(folder / 'model')]
    c.write(folder / 'fit_started.json', {'started_at_utc': c.now(), 'command': command,
            'prepared': c.record(prepared), 'all_input_arrays': identities, 'automatic_retry': False})
    tick = time.monotonic()
    with (folder / 'trainer.log').open('x', encoding='utf8') as log:
        completed = subprocess.run(command, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    c.write(folder / 'execution.json', {'exit_code': completed.returncode, 'failure': None if completed.returncode == 0 else 'trainer subprocess failed',
            'elapsed_seconds': time.monotonic()-tick, 'command': command, 'automatic_retry': False})
    c.need(completed.returncode == 0, 'Trainer failed; no automatic retry')
    training = c.read(folder / 'model/training.json')
    rows, n = len(arrays['train_y_million']), len(arrays['query_num'])
    c.need(training['status'] == 'completed' and training['configuration'] == baseline['configuration'], 'Original R63 config differs')
    c.need(training['source_sha256'] == c.TRAINER_SHA and training['architecture_source_sha256'] == old.ARCHITECTURE_SHA,
           'Trainer/architecture differs')
    c.need(training['input_sha256'] == c.sha(prepared) and training['fit_rows'] == rows and training['query_rows'] == n,
           'Prepared input/row contract differs')
    c.need(training['numeric_count'] == 144 and training['cardinalities'] == arrays['cardinalities'].tolist(), 'Encoding dimensions differ')
    c.need(training['query_labels_available'] is False and training['query_labels_used'] is False
           and training['saved_model_reload_exact'] is True and training['reload_rows'] == n, 'Query/reload contract differs')
    c.need([e['epoch'] for e in training['epochs']] == list(range(1, 31)), 'Incomplete fixed epochs')
    c.need(training['sample_weights_batch_renormalized'] is False and training['price_or_ratio_clipping'] is False,
           'Weight normalization or clipping changed')
    for k in ('train_price_factor_million', 'query_price_factor_million', 'train_sample_weight'):
        c.need(training[k + '_sha256'] == identities[k]['values_sha256'], 'Factor/weight identity differs')
    ratio = arrays['train_y_million'].astype(np.float64) / arrays['train_price_factor_million'].astype(np.float64)
    median = float(np.median(ratio)); bias = median + float(np.log(-np.expm1(-median)))
    c.need(training['initial_training_median_ratio'] == median and training['initial_head_bias'] == bias, 'Arm-specific initial bias differs')
    native = np.load(folder / 'model/predictions_twd.npy', allow_pickle=False)
    positive(native, n, 'float64', 'NN prediction')
    info = {'prepared': c.record(prepared), 'training': c.record(folder / 'model/training.json'),
            'model': c.record(folder / 'model/model.pt'), 'execution': c.record(folder / 'execution.json'),
            'all_input_arrays': identities, 'fit_seconds': training['fit_seconds'],
            'saved_model_reload_exact': True, 'reload_rows': n, 'original_R63_configuration_exact': True,
            'native_prediction': {**c.record(folder / 'model/predictions_twd.npy'), **c.identity(native)}}
    return native, info


def stage_run(stage, request_path):
    request = check_launch(request_path)
    c.need(OUT.is_dir() and c.read(OUT/'started.json')['request'] == c.record(request_path), 'Stage must belong to this serial parent')
    folder = OUT / stage
    c.need(not folder.exists(), 'Stage exists; preserve previous run')
    folder.mkdir(); tick = time.monotonic(); code = 1; failure = None
    inputs = {}; sources = request['source_sha256']; before = c.protected_hashes()
    try:
        def checked(name, digest):
            name = c.rel(name)
            c.need(name != c.RAW and not name.startswith(('data_cache/test', 'data_cache/valid')), 'Forbidden raw/query-price input')
            if name.endswith('.npz'):
                c.need(name == f'outputs/round63/{stage}/prepared.npz', 'Another stage prepared input forbidden')
            if name.startswith('outputs/round100_inputs/') and name.endswith('.npy'):
                c.need(name.startswith(f'outputs/round100_inputs/{stage}/'), 'Another stage fit labels forbidden')
            c.need(name not in inputs or inputs[name] == digest, 'Conflicting original input identity')
            if name not in inputs:
                c.need(c.sha(name) == digest, 'Original input changed: ' + name)
                inputs[name] = digest
            return c.safe(name)
        history = {p: c.read(checked(p, request['input_authority_sha256'][p])) for p in HISTORY}
        p48, p79 = (history[p] for p in HISTORY[:2])
        records = {r: history[f'outputs/round{r}/results.json'] for r in (48,54,60,63,71,79,83)}
        c.need(all(r['checked_stages'] == list(c.STAGES) for r in records.values()), 'Historical stages incomplete')
        c.need(p48['split_contract'] == p79['split_contract'] and p48['splits'] == p79['splits'], 'Historical split differs')
        c.verify_hashes(p48['source_sha256']); c.verify_hashes(p79['source_sha256'])
        labels = c.read(checked(c.FIT_LABELS, request['input_authority_sha256'][c.FIT_LABELS]))
        audit = c.read(checked(c.X_AUDIT, request['input_authority_sha256'][c.X_AUDIT]))
        c.need(labels['status'] == 'fit_only_labels_sealed' and labels['plan_sha256'] == c.PLAN_SHA, 'Fit-label export authority differs')
        c.need(audit['status'] == 'completed_pure_X_support_audit' and audit['fit_predict_score_calls'] == 0
               and audit['prior_build_calls'] == 0 and audit['labels_rawpricepickle_NPZ_models_loaded'] is False
               and audit['source_and_inputs_unchanged'] is True, 'Pure-X support audit incomplete')
        spec = labels['stages'][stage]; ax = audit['stages'][stage]
        start, end, nf, nq = c.SPLITS[stage]
        c.need((spec['cutoff'], spec['query_end'], spec['fit_rows'], spec['query_rows']) == (start,end,nf,nq), 'Fit-label split differs')
        def arr(rec):
            checked(rec['path'], rec['sha256'])
            return c.load_array(rec)
        def historical(r, suffix):
            name = f'outputs/round{r}/{stage}/{suffix}'
            return checked(name, records[r]['stages'][stage]['artifact_sha256'][name])
        def vec(path):
            a = np.load(path, allow_pickle=False); positive(a,nq,'float64','saved TWD',zero=True); return a
        fi = arr(spec['arrays']['fit_indices']); rawfi = arr(spec['arrays']['fit_source_cache_indices'])
        yfit = arr(spec['arrays']['fit_y']); positive(yfit,nf,'float64','full-precision own-stage fit labels')
        c.need(spec['arrays']['fit_y']['path'] == f'outputs/round100_inputs/{stage}/fit_y.npy', 'Stage fit label path differs')
        qi = np.load(historical(63,'query_indices.npy'),allow_pickle=False)
        c.need(fi.dtype == qi.dtype == np.int64 and fi.shape == (nf,) and qi.shape == (nq,), 'Canonical split schema')
        c.need((np.diff(fi)>0).all() and (np.diff(qi)>0).all() and not np.intersect1d(fi,qi).size, 'Canonical order/overlap')
        for k,a in [('fit',fi),('query',qi)]:
            c.need(c.values_sha(a) == p79['split_contract'][stage][k+'_indices_content_sha256'], 'Canonical content differs')
            for r in (48,54,60,63,71,79,83):
                c.exact(a,np.load(historical(r,k+'_indices.npy'),allow_pickle=False),'Historical canonical order')
        for key, path in [('features',c.FEATURES),('canonical_mapping',c.CANONICAL)]:
            c.need(labels[key]['path'] == path and labels[key]['sha256'] == p79['input_sha256'][path], 'Pure-X authority differs')
            checked(path, labels[key]['sha256'])
        mapping=np.load(c.safe(c.CANONICAL),allow_pickle=False)
        c.need(mapping.dtype==np.int64 and mapping.shape==(455255,) and (np.diff(mapping)>0).all(), 'Canonical mapping schema')
        c.exact(rawfi,mapping[fi],'Raw fit-label mapping'); rawqi=mapping[qi]
        Xraw=pd.read_pickle(c.safe(c.FEATURES))
        c.need(len(Xraw)==455784 and '總價元' not in Xraw and Xraw.index.equals(pd.RangeIndex(len(Xraw))), 'Raw feature cache schema')
        X=Xraw.iloc[rawfi].reset_index(drop=True); Xq=Xraw.iloc[rawqi].reset_index(drop=True); del Xraw
        frames={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}
        c.need(frames==records[79]['stages'][stage]['original_frame_sha256']==records[63]['stages'][stage]['original_frame_sha256'], 'Outer frame identity')
        day=lambda x:(pd.Timestamp(x)-pd.Timestamp('2000-01-01')).days
        c.need(np.isfinite(X.trade_day_index).all() and (X.trade_day_index<day(start)).all()
               and Xq.trade_day_index.between(day(start),day(end)).all(), 'Outer time boundary differs')
        c.exact(arr(ax['arrays']['fit_indices']),fi,'Pure-X canonical indices')
        folds=arr(ax['arrays']['fold_ids']); eligible=arr(ax['arrays']['eligible_mask'])
        c.need(eligible.dtype==bool and eligible.shape==(nf,) and eligible.any()
               and int(eligible.sum())==ax['eligible_rows'] and ax['fit_rows']==nf, 'Pure-X eligible mask')
        c.exact(folds,np.load(historical(60,'prior_fold_ids.npy'),allow_pickle=False),'Original group folds')
        b=p48['bindings'][stage]; b79=p79['bindings'][stage]
        prepared=historical(63,'prepared.npz'); baseline=c.read(historical(63,'model/training.json'))
        c.need(baseline['input_sha256']==c.sha(prepared) and baseline['source_sha256']==c.TRAINER_SHA, 'Original R63 authority')
        with np.load(prepared,allow_pickle=False) as z:
            c.need(set(z.files)==set(c.ARRAY_NAMES),'Exactly nine original trainer arrays')
            original={k:z[k] for k in c.ARRAY_NAMES}
        c.exact(original['train_y_million'],np.asarray(yfit/1e6,dtype=np.float32),'Original fit labels alignment')
        card=original['cardinalities']; c.need(card.dtype==np.int64 and card.shape==(23,), 'Original cardinalities')
        for prefix,rows in [('train',nf),('query',nq)]:
            n,a=original[prefix+'_num'],original[prefix+'_cat']
            c.need(n.dtype==np.float32 and n.shape==(rows,144) and np.isfinite(n).all(), 'Original numeric matrix')
            c.need(a.dtype==np.int64 and a.shape==(rows,23) and (a>=0).all() and (a<card).all(), 'Original category matrix')
        # Reuse the original full-fit encoder. R6 geography feeds NN; R39 feeds R/S.
        Gfull=pd.read_pickle(checked(b['original_geo'],p48['input_sha256'][b['original_geo']]))
        c.need(len(Gfull)==len(mapping) and Gfull.index.equals(pd.RangeIndex(len(mapping))), 'R6 geography alignment')
        enc=c.read(checked(b['R47_encoder'],p48['input_sha256'][b['R47_encoder']]))
        c.need(enc['fit_rows']==nf and enc['query_used_for_state'] is False and enc['target_used_for_state'] is False,
               'Encoder full-fit provenance')
        for prefix,frame,ids in [('train',X,fi),('query',Xq,qi)]:
            augmented=geo_encoder.augment_features(frame,Gfull.iloc[ids].reset_index(drop=True))
            numeric,categorical=encoder.encode(enc,augmented)
            c.exact(numeric,original[prefix+'_num'],'Original encoded numeric replay')
            c.exact(categorical,original[prefix+'_cat'],'Original encoded categorical replay')
        del Gfull, augmented, numeric, categorical
        old_prior=np.load(historical(60,'train_oof_unit_prior.npy'),allow_pickle=False)
        query_prior=np.load(historical(60,'query_unit_prior.npy'),allow_pickle=False)
        prior_state=c.read(historical(60,'price_prior_state.json')); prior_meta=c.read(historical(60,'prior_preparation.json'))
        pe=prior_meta['prior_evidence']
        c.need(pe['fit_target_sha256']==c.values_sha(yfit) and pe['source_indices_sha256']==c.values_sha(fi)
               and pe['fold_ids_sha256']==c.values_sha(folds) and pe['oof_prior_sha256']==c.values_sha(old_prior), 'Original OOF fit provenance')
        c.exact(price_prior.predict_prior(prior_state,Xq),query_prior,'Original full-fit query prior replay')
        area=area_helper.effective_area(X); qarea=area_helper.effective_area(Xq)
        for key,A,P in [('train_price_factor_million',area,old_prior),('query_price_factor_million',qarea,query_prior)]:
            c.exact(np.asarray(A*P/1e6,dtype=np.float32),original[key],'Original price-factor replay')
        weights,valid,date_report=price_prior._weights(price_prior._dates(X))
        c.exact(weights.astype(np.float32),original['train_sample_weight'],'Original full-fit date weight replay')
        c.need(c.values_sha(original['train_sample_weight'])==ax['original_date_weight32_values_sha256'], 'Pure-X date weight identity')
        prefix_dir=folder/'prefix_states';prefix_dir.mkdir()
        def save_prefix_state(quarter_day,fold,state):
            path=prefix_dir/f'q{int(quarter_day)}_fold{int(fold)}.json'
            c.write(path,state)
            return c.record(path)
        temporal,kept,evidence=helper.build_temporal_train_prior(X,yfit,fi,folds,eligible,state_sink=save_prefix_state)
        c.exact(kept,np.flatnonzero(eligible).astype(np.int64),'Temporal prior eligible order')
        c.write(folder/'temporal_prior_evidence.json',evidence)
        prior_record=c.save_array(folder/'temporal_train_prior.npy',temporal)
        arms_arrays=masked_arrays(original,eligible,temporal,area)
        common_before={k:c.identity(v) for k,v in original.items()}
        c.write(folder/'preparation.json',{'status':'matched_prior_arms_prepared','source_prepared':c.record(prepared),
                'original_arrays':common_before,'kept_positions':c.identity(kept),'eligible_rows':len(kept),
                'only_arm_array_difference':'train_price_factor_million','date_weights_renormalized':False,
                'encoder_query_prior_full_outer_fit_scope':True,'query_labels_available':False,
                'fit_labels':spec['arrays']['fit_y'],'pure_X_audit':c.record(c.X_AUDIT),
                'temporal_prior_evidence':c.record(folder/'temporal_prior_evidence.json'),
                'date_weight_evidence':date_report,'old_OOF_identity_verified_not_recomputed':True})
        def bound(name): return checked(name,p48['input_sha256'][name])
        with bound(b['reference']).open('rb') as f: reference=pickle.load(f)
        with bound(b['state']).open('rb') as f: spatial_state=pickle.load(f)
        summary=c.read(bound(b['state_summary']))
        c.need(spatial_state['state_summary']==summary and summary['fit_X_sha256']==frames['fit_X']
               and not summary['query_used_in_build'] and reference['fit_report']['fitting_rows']==nf,'Original R/S fit provenance')
        c.need(spatial_state['parameters']==old.spatial.parameters(), 'Original spatial parameters')
        geo=pd.read_pickle(bound(b['query_geo']))
        c.need(geo.index.equals(Xq.index) and old.spatial.frame_sha(geo)==b['manifest']['features']['query']['frame_sha256'], 'R39 query geography')
        state_before=hashlib.sha256(pickle.dumps((reference,spatial_state),protocol=5)).hexdigest()
        fixed={'area':vec(historical(54,'model/predictions_twd.npy')),'lgb':vec(bound(b['old_unit'])),
               'H':vec(checked(b79['paths']['log'],p79['input_sha256'][b79['paths']['log']])),'U':vec(historical(79,'unit_spatial.npy'))}
        native_old=vec(historical(63,'model/predictions_twd.npy'))
        predictions={'v22':vec(historical(79,'predictions.npy')),'v24':vec(historical(83,'predictions.npy'))}
        layers={};arm_records={}
        def chain(native,name):
            base=.75*(.5*fixed['area']+.5*native)+.25*fixed['lgb']
            ref,diag,_=old.original.predict_reference(reference,Xq,base,variant='trend',return_diagnostics=True)
            spatial,detail,_=old.spatial.predict(spatial_state,Xq,geo,base,ref,diag)
            fallback=~detail['applied'].to_numpy(dtype=bool)
            c.exact(spatial[fallback],ref[fallback],'Spatial fallback')
            layer={'native':native,'base':base,'reference':ref,'spatial':spatial}
            if name=='v22':
                selected_nn_path=b79['paths']['nn']
                c.exact(spatial,vec(checked(selected_nn_path,p79['input_sha256'][selected_nn_path])),
                        'Original R79 selected NN branch')
                for field,oldname in [('base','base'),('reference','reference'),('spatial','nn_spatial')]:
                    c.exact(layer[field],vec(historical(71,f'seed42/{oldname}.npy')),'Original R71 '+field)
                for fname,frame in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
                    with historical(71,f'seed42/{fname}.pkl').open('rb') as f:
                        pd.testing.assert_frame_equal(frame,pickle.load(f),check_exact=True)
            save=folder/name;save.mkdir(exist_ok=True)
            layers[name]={k:c.save_array(save/(k+'.npy'),a) for k,a in layer.items()}
            for fname,frame in [('reference_diagnostics',diag),('spatial_diagnostics',detail)]:
                with (save/(fname+'.pkl')).open('xb') as f:pickle.dump(frame,f,protocol=5)
                with (save/(fname+'.pkl')).open('rb') as f:pd.testing.assert_frame_equal(frame,pickle.load(f),check_exact=True)
            return np.median(np.column_stack([spatial,fixed['H'],fixed['U']]),axis=1)
        c.exact(chain(native_old,'v22'),predictions['v22'],'Original full v22 median')
        for arm in ('C','T'):
            c.verify_hashes(sources);c.verify_hashes(inputs)
            native,arm_records[arm]=train_arm(folder/arm,arms_arrays[arm],baseline)
            predictions[arm]=chain(native,arm)
            print(json.dumps({'stage':stage,'arm':arm,'fit_seconds':arm_records[arm]['fit_seconds'],'full_query_reload_exact':True}),flush=True)
        c.need(common_before=={k:c.identity(v) for k,v in original.items()}, 'Original NPZ arrays mutated')
        c.need(frames=={'fit_X':old.spatial.frame_sha(X),'query_X':old.spatial.frame_sha(Xq)}, 'Original frames mutated')
        c.need(state_before==hashlib.sha256(pickle.dumps((reference,spatial_state),protocol=5)).hexdigest(), 'Original R/S state mutated')
        c.verify_hashes(inputs);c.verify_hashes(sources)
        c.need(before==c.protected_hashes(), 'Protected artifacts changed')
        seal={'schema_version':1,'status':'temporal_prior_stage_predictions_sealed','stage':stage,
              'cutoff':start,'query_end':end,'fit_rows':nf,'query_rows':nq,'eligible_rows':len(kept),
              'new_NN_fits':2,'new_native_tree_fits':0,'query_scores_computed':False,'query_labels_available':False,
              'raw_price_pickle_opened':False,'other_stage_fit_labels_opened':False,'all_query_reload_exact':True,
              'original_v22_full_chain_and_diagnostics_exact':True,'original_R63_configuration_exact':True,
              'plan_sha256':c.PLAN_SHA,'request':c.record(request_path),'source_sha256':sources,
              'fit_label_manifest':c.record(c.FIT_LABELS),'pure_X_audit':c.record(c.X_AUDIT),
              'features':labels['features'],'canonical_mapping':labels['canonical_mapping'],'raw_train':labels['raw_train'],
              'original_input_sha256':inputs,'protected_before':before,'protected_after':c.protected_hashes(),
              'arms':arm_records,'layers':layers,'formula':FORMULA,
              'temporal_prior':prior_record,'temporal_prior_evidence':c.record(folder/'temporal_prior_evidence.json'),
              'preparation':c.record(folder/'preparation.json'),'fit_seconds':sum(a['fit_seconds'] for a in arm_records.values()),
              'elapsed_seconds':time.monotonic()-tick}
        for name,a in [('fit_indices',fi),('query_indices',qi),('fit_source_cache_indices',rawfi),
                       ('query_source_cache_indices',rawqi),('eligible_positions',kept)]:
            seal[name]=c.save_array(folder/(name+'.npy'),a)
        seal['fixed_branches']={k:c.save_array(folder/(k+'.npy'),a) for k,a in fixed.items()}
        seal['predictions']={k:c.save_array(folder/('prediction_'+k+'.npy'),a) for k,a in predictions.items()}
        seal['artifact_sha256']={c.rel(p):c.sha(p) for p in sorted(folder.rglob('*')) if p.is_file()}
        c.write(folder/'stage_seal.json',seal);code=0
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        c.write(folder/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
                'stage':stage,'query_scores_computed':False,'automatic_retry':False})


def pipeline(request_path):
    request=check_launch(request_path)
    c.need(not OUT.exists(),'Existing R100 output; preserve previous evidence')
    OUT.mkdir();tick=time.monotonic();code=1;failure=None
    c.write(OUT/'started.json',{'started_at_utc':c.now(),'request':c.record(request_path),'plan_sha256':c.PLAN_SHA,'automatic_retry':False})
    try:
        stages={}
        for stage in c.STAGES:
            command=[sys.executable,'-u','-B','-X','utf8',str(Path(__file__).resolve()),'--stage',stage,'--request',str(c.safe(request_path))]
            with (OUT/(stage+'_process.log')).open('x',encoding='utf8') as f:
                done=subprocess.run(command,cwd=c.ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
            c.need(done.returncode==0,'Stage failed; stop without automatic retry: '+stage)
            folder=OUT/stage;st=c.read(folder/'stage_seal.json');ex=c.read(folder/'execution.json')
            c.need(st['new_NN_fits']==2 and st['all_query_reload_exact'] and ex['exit_code']==0 and ex['failure'] is None,'Incomplete stage')
            stages[stage]={'stage_manifest':c.record(folder/'stage_seal.json'),'execution':c.record(folder/'execution.json'),
                           'predictions':st['predictions'],'fit_seconds':st['fit_seconds']}
        c.need(request['protected_before']==c.protected_hashes(),'Protected artifacts changed')
        c.verify_hashes(request['frozen_source_and_plan']);c.verify_hashes(request['input_authority_sha256'])
        labels=c.read(c.FIT_LABELS)
        seal={'schema_version':1,'status':'all_temporal_prior_predictions_sealed','stage_order':list(c.STAGES),
              'new_NN_fits':6,'new_native_tree_fits':0,'query_scores_computed':False,'query_labels_available':False,
              'raw_price_pickle_opened':False,'other_stage_fit_labels_opened':False,'all_query_reload_exact':True,
              'plan_sha256':c.PLAN_SHA,'source_sha256':request['source_sha256'],'request':c.record(request_path),
              'fit_label_manifest':c.record(c.FIT_LABELS),'pure_X_audit':c.record(c.X_AUDIT),
              'features':labels['features'],'canonical_mapping':labels['canonical_mapping'],'raw_train':labels['raw_train'],
              'protected_before':request['protected_before'],'protected_after':c.protected_hashes(),'stages':stages,
              'fit_seconds':sum(s['fit_seconds'] for s in stages.values()),'elapsed_seconds':time.monotonic()-tick,
              'new_CSV':False,'selected_submission':'v22','Private_score':None}
        c.write(OUT/'sealed_manifest.json',seal);code=0
        print(json.dumps({'status':seal['status'],'new_NN_fits':6,'fit_seconds':seal['fit_seconds'],'elapsed_seconds':seal['elapsed_seconds']}),flush=True)
    except Exception:
        failure=traceback.format_exc();raise
    finally:
        c.write(OUT/'execution.json',{'exit_code':code,'failure':failure,'elapsed_seconds':time.monotonic()-tick,
                'source_sha256':c.sha(__file__),'automatic_retry':False,'query_scores_computed':False})


def self_check():
    tick=time.monotonic();checks=[]
    x=np.arange(20,dtype=np.float32).reshape(5,4)
    a={'train_num':x,'train_cat':np.zeros((5,2),dtype=np.int64),'train_y_million':np.arange(1,6,dtype=np.float32),
       'query_num':x[:2].copy(),'query_cat':np.zeros((2,2),dtype=np.int64),'cardinalities':np.array([1,1],dtype=np.int64),
       'train_price_factor_million':np.ones(5,dtype=np.float32),'query_price_factor_million':np.ones(2,dtype=np.float32),
       'train_sample_weight':np.array([.1,.2,.3,1.,3.4],dtype=np.float32)}
    before={k:c.identity(v) for k,v in a.items()};mask=np.array([False,True,False,True,True]);prior=np.array([10000.,20000.,40000.]);area=np.full(5,100.)
    arms=masked_arrays(a,mask,prior,area)
    for k in c.ARRAY_NAMES:
        if k!='train_price_factor_million':
            c.exact(arms['C'][k],arms['T'][k],k);checks.append('shared_'+k)
    c.exact(arms['C']['train_price_factor_million'],a['train_price_factor_million'][mask],'C factor original subset');checks.append('C_factor_subset')
    c.exact(arms['T']['train_price_factor_million'],np.array([1.,2.,4.],dtype=np.float32),'T factor float64 then float32');checks.append('T_factor_formula')
    c.need(before=={k:c.identity(v) for k,v in a.items()},'Input mutated');checks.append('caller_inputs_unchanged')
    for label,badmask,badprior in [('empty',np.zeros(5,dtype=bool),np.array([],dtype=np.float64)),
                                 ('maskdtype',mask.astype(np.int64),prior),('nonpositive',mask,np.array([1.,0.,2.]))]:
        try:masked_arrays(a,badmask,badprior,area)
        except ValueError:checks.append('reject_'+label)
        else:raise AssertionError('Invalid synthetic input accepted')
    return {'status':'passed','check_count':len(checks),'checks':checks,'source_sha256':c.sha(__file__),
            'real_data_read':False,'new_real_native_fits':0,'query_scores_computed':False,'GPU_calls':0,
            'elapsed_seconds':time.monotonic()-tick}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);mode=p.add_mutually_exclusive_group(required=True)
    mode.add_argument('--run',action='store_true');mode.add_argument('--stage',choices=c.STAGES);mode.add_argument('--self-check',action='store_true')
    p.add_argument('--request',default=REQUEST);p.add_argument('--receipt')
    args=p.parse_args()
    if args.self_check:
        result=self_check()
        if args.receipt:c.write(args.receipt,result)
        print(json.dumps(result,ensure_ascii=False))
    elif args.stage:stage_run(args.stage,args.request)
    else:pipeline(args.request)
