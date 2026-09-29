"""Exactly two R104 pre-auxiliary teachers, serial, fit-label-only process.

The immutable R48 area trainer runs in the isolated Torch interpreter. Unit
LightGBM uses original X87 and original v1 3999-tree parameters. No price
pickle, query labels, prior network, reference, spatial model or scoring here.
"""
from __future__ import annotations
import argparse
import ast
import json
import subprocess
import time
import traceback
import numpy as np
import pandas as pd
import lightgbm as lgb
import round100_contract as c
import round104_teacher_preparation as p
import train_lightgbm as light


def expected_area_config():
    tree = ast.parse(c.safe(p.ARCHITECTURE).read_text(encoding='utf-8-sig'))
    node = next(n for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(v, ast.Name) and v.id == 'CONFIG' for v in n.targets))
    cfg = ast.literal_eval(node.value)
    cfg.update(prediction_head='Softplus unit price; unit * area_factor yields million TWD',
               output_units='1e4 TWD per square metre of effective area A',
               area_formula='finite positive building_area else finite positive parking_area else 1',
               area_factor_formula='A / 100',
               loss_formula='abs(unit_output * train_area_factor - train_y_million).mean()',
               initial_bias='inverseSoftplus(median_fit(train_y_million / train_area_factor))',
               extra_sample_weights=False, price_or_unit_clipping=False)
    return cfg


def lgb_parameters():
    c.need(c.sha(p.CONFIG) == p.CONFIG_SHA, 'Original v1 config SHA differs')
    component = next(a for a in c.read(p.CONFIG)['components'] if a['family'] == 'lightgbm')
    c.need(component['target_mode'] == 'unit' and component['iterations'] == 3999, 'Original LGB identity differs')
    params = {**component['parameters'], 'n_estimators': component['iterations'], 'n_jobs': 4}
    c.need(params == light.candidate_params(63, 3999, 4), 'Original LGB parameter contract differs')
    return params


def parameters():
    return {'stage': 'pre_auxiliary', 'origin': p.ORIGIN, 'query_end': p.QUERY_END,
            'model_order': ['area', 'lgb'], 'new_NN_fits': 1, 'new_native_tree_fits': 1,
            'area_configuration': expected_area_config(), 'lgb_parameters': lgb_parameters(),
            'geography': p.GEO_LIMIT, 'query_prices_read': False, 'raw_price_pickle_opened': False,
            'retry': False, 'overwrite': False}


def add_record(inputs, rec):
    p.check_record(rec)
    old = inputs.get(rec['path'])
    c.need(old is None or old == rec['sha256'], 'Conflicting SHA')
    inputs[rec['path']] = rec['sha256']


def freeze(path, preparation, labels):
    """Metadata/file-hash preflight; no arrays, targets or models loaded."""
    inputs = {}; prep = c.read(preparation); lab = c.read(labels)
    c.need(prep['status'] == 'pure_X_teacher_prepared' and prep['plan_sha256'] == p.PLAN_SHA and
           lab['status'] == 'teacher_fit_labels_sealed' and lab['plan_sha256'] == p.PLAN_SHA,
           'Completed teacher preparation and fit labels required')
    c.need(lab['preparation'] == c.record(preparation) and lab['fit_rows'] == prep['fit_rows'], 'Fit labels preparation differs')
    for name in ('fit_indices', 'fit_source_cache_indices'):
        c.need(lab[name] == prep['indices'][name], 'Fit-label index identity differs')
    c.need(prep['source_sha256'] == lab['source_sha256'] == p.source_hashes(), 'Source identity differs')
    c.verify_hashes(prep['source_sha256'])
    for rec in [c.record(preparation), c.record(labels), prep['prepared'], lab['fit_y'],
                *prep['indices'].values(), *prep['frames'].values(), *prep['states'].values()]:
        add_record(inputs, rec)
    for manifest in (preparation, labels):
        ex = c.safe(manifest).parent / 'execution.json'; receipt = c.read(ex)
        c.need(receipt['exit_code'] == 0 and receipt['failure'] is None, 'Preparation process incomplete')
        add_record(inputs, c.record(ex))
    inputs[p.PLAN] = p.PLAN_SHA; inputs[p.CONFIG] = p.CONFIG_SHA
    c.verify_hashes(inputs)
    protected = c.protected_hashes()
    c.need(protected == prep['protected_before'] == prep['protected_after'] == lab['protected_after'], 'Protected files changed')
    c.safe(path).parent.mkdir(parents=True, exist_ok=True)
    c.write(path, {'schema_version': 1, 'status': 'two_teacher_fit_request_frozen',
                   'created_at_utc': c.now(), 'plan_sha256': p.PLAN_SHA, 'parameters': parameters(),
                   'preparation': c.record(preparation), 'fit_label_manifest': c.record(labels),
                   'input_sha256': inputs, 'source_sha256': p.source_hashes(), 'protected_before': protected,
                   'raw_train_declared_only': prep['raw_train'], 'arrays_or_prices_deserialized': False})


def fit_lgb(Xfit, Xquery, y, category, out):
    """Train exactly one original unit-L1 model with frozen fit-only categories."""
    out.mkdir(exist_ok=False)
    frames = {'train': Xfit, 'query': Xquery}
    encoded, maps, names, catcols = light.encode_features(frames, category['original_categorical_columns'], category['category_maps'])
    c.need(maps == category['category_maps'] and names == category['safe_names'] and
           catcols == category['categorical_columns'] and category['fit_rows'] == len(y), 'LGB categories differ')
    factor = light.price_factor(Xfit, 'unit'); qfactor = light.price_factor(Xquery, 'unit')
    weights = factor / 1e6
    params = lgb_parameters(); model = lgb.LGBMRegressor(**params)
    t = time.monotonic()
    model.fit(encoded['train'], y / factor, sample_weight=weights, categorical_feature=catcols)
    fit_seconds = time.monotonic() - t
    c.need(model.booster_.current_iteration() == 3999, 'LightGBM iteration count differs')
    prediction = np.maximum(np.asarray(model.predict(encoded['query'], num_threads=4), dtype=np.float64) * qfactor, 0.)
    model.booster_.save_model(str(out / 'model.txt'))
    c.write(out / 'category_state.json', category)
    re_category = c.read(out / 'category_state.json')
    re_encoded, _, _, _ = light.encode_features(frames, re_category['original_categorical_columns'], re_category['category_maps'])
    for role in frames: pd.testing.assert_frame_equal(encoded[role], re_encoded[role], check_exact=True)
    re_model = lgb.Booster(model_file=str(out / 'model.txt'))
    re_prediction = np.maximum(np.asarray(re_model.predict(re_encoded['query'], num_threads=4), dtype=np.float64) * qfactor, 0.)
    c.exact(prediction, re_prediction, 'LightGBM saved model full query differs')
    c.need(prediction.shape == (len(Xquery),) and np.isfinite(prediction).all() and (prediction >= 0).all(), 'LGB prediction invalid')
    rec = c.save_array(out / 'predictions_twd.npy', prediction)
    metadata = {'status': 'completed', 'parameters': params, 'target_mode': 'unit', 'iterations': 3999,
                'fit_rows': len(y), 'query_rows': len(Xquery), 'feature_count': 87,
                'categorical_count': len(catcols), 'fit_seconds': fit_seconds,
                'target_identity': c.identity(y / factor), 'sample_weight_identity': c.identity(weights),
                'sample_weight_formula': 'effective_area / 100', 'query_labels_available': False,
                'geography_features_used': False, 'eval_set_used': False, 'saved_model_reload_exact': True,
                'reload_rows': len(prediction), 'prediction': rec, 'model': c.record(out / 'model.txt'),
                'category_state': c.record(out / 'category_state.json'), 'lightgbm_version': lgb.__version__}
    c.write(out / 'training.json', metadata)
    return rec, metadata


def check_area_training(meta, arrays, data_path):
    c.need(meta['status'] == 'completed' and meta['configuration'] == expected_area_config(), 'Area configuration differs')
    c.need(meta['source_sha256'] == p.TRAINER_SHA and meta['architecture_source_sha256'] == p.ARCHITECTURE_SHA and
           meta['input_sha256'] == c.sha(data_path), 'Area trainer source or NPZ differs')
    c.need(meta['fit_rows'] == len(arrays['train_num']) and meta['query_rows'] == len(arrays['query_num']) and
           meta['numeric_count'] == arrays['train_num'].shape[1] and
           meta['cardinalities'] == arrays['cardinalities'].tolist(), 'Area dimensions differ')
    c.need(meta['saved_model_reload_exact'] and meta['reload_rows'] == len(arrays['query_num']) and
           meta['query_labels_available'] is False and meta['query_labels_used'] is False,
           'Area query boundary or reload failed')
    c.need([e['epoch'] for e in meta['epochs']] == list(range(1, 31)), 'Area epochs differ')
    for key in ('train_area_factor', 'query_area_factor'):
        c.need(meta[key + '_sha256'] == c.values_sha(arrays[key]), 'Area factor identity differs')
    median = float(np.median(arrays['train_y_million'].astype(np.float64) / arrays['train_area_factor'].astype(np.float64)))
    c.need(meta['initial_training_median_unit_1e4_twd_per_m2'] == median and
           meta['initial_head_bias'] == median + float(np.log(-np.expm1(-median))), 'Area initialization differs')


def run(request, out):
    r = c.read(request)
    c.need(r['status'] == 'two_teacher_fit_request_frozen' and r['plan_sha256'] == p.PLAN_SHA and
           r['parameters'] == parameters() and r['source_sha256'] == p.source_hashes(), 'Fit request differs')
    c.verify_hashes(r['source_sha256']); c.verify_hashes(r['input_sha256'])
    c.need(c.protected_hashes() == r['protected_before'], 'Protected files changed')
    out = c.safe(out); out.mkdir(parents=True, exist_ok=False)
    t = time.monotonic(); status = 'failed'; error = None; fits_started = 0; fits_completed = 0
    try:
        m, indices, frames, arrays = p.load_prepared(r['preparation']['path'])
        labels = c.read(r['fit_label_manifest']['path']); y = c.load_array(labels['fit_y'])
        c.need(y.dtype == np.float64 and y.shape == (m['fit_rows'],) and np.isfinite(y).all() and (y > 0).all(), 'Fit target identity differs')
        c.need(labels['preparation'] == r['preparation'] and labels['fit_indices'] == m['indices']['fit_indices'] and
               labels['fit_source_cache_indices'] == m['indices']['fit_source_cache_indices'], 'Fit target row binding differs')
        arrays['train_y_million'] = (y / 1e6).astype(np.float32)
        c.need(np.isfinite(arrays['train_y_million']).all() and (arrays['train_y_million'] > 0).all(), 'Float32 fit target invalid')
        data = p.save_npz(out / 'area_training.npz', arrays)
        process = out / 'area_process'; process.mkdir()
        cmd = [str(c.ROOT / '.venv_torch/Scripts/python.exe'), '-u', '-B', '-X', 'utf8', str(c.safe(p.TRAINER)),
               '--data', str(c.safe(data['path'])), '--out', str(out / 'area')]
        c.write(process / 'started.json', {'created_at_utc': c.now(), 'command': cmd, 'request': c.record(request),
                'trainer_source_sha256': p.TRAINER_SHA, 'input': data})
        fits_started += 1; started = time.monotonic()
        with (process / 'stdout.log').open('x', encoding='utf8') as log:
            completed = subprocess.run(cmd, cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
        c.write(process / 'execution.json', {'exit_code': completed.returncode, 'elapsed_seconds': time.monotonic() - started,
                'request': c.record(request), 'retry': False})
        c.need(completed.returncode == 0, 'Area trainer failed; no automatic retry')
        area_meta = c.read(out / 'area/training.json'); check_area_training(area_meta, arrays, data['path'])
        area = np.load(out / 'area/predictions_twd.npy', allow_pickle=False)
        c.need(area.dtype == np.float64 and area.shape == (m['query_rows'],) and np.isfinite(area).all() and (area > 0).all(), 'Area predictions invalid')
        ar = c.save_array(out / 'area_predictions.npy', area); fits_completed += 1
        del arrays
        category = c.read(m['states']['lgb_category_state']['path'])
        fits_started += 1
        lr, lgb_meta = fit_lgb(frames['Xfit'], frames['Xquery'], y, category, out / 'lgb')
        lp = c.load_array(lr); lr = c.save_array(out / 'lgb_predictions.npy', lp); fits_completed += 1
        teachers = {'area': {'training': c.record(out / 'area/training.json'), 'model': c.record(out / 'area/model.pt'),
                            'execution': c.record(process / 'execution.json'), 'fit_seconds': area_meta['fit_seconds'],
                            'saved_model_reload_exact': True, 'reload_rows': m['query_rows']},
                    'lgb': {'training': c.record(out / 'lgb/training.json'), 'model': c.record(out / 'lgb/model.txt'),
                            'fit_seconds': lgb_meta['fit_seconds'], 'saved_model_reload_exact': True, 'reload_rows': m['query_rows']}}
        seal = {'schema_version': 1, 'status': 'pre_auxiliary_teachers_sealed', 'stage': 'pre_auxiliary',
                'origin': p.ORIGIN, 'cutoff': p.ORIGIN, 'query_end': p.QUERY_END,
                'fit_rows': m['fit_rows'], 'query_rows': m['query_rows'], 'plan_sha256': p.PLAN_SHA,
                'features': m['features'], 'canonical_mapping': m['canonical_mapping'], **m['indices'],
                'native_predictions': {'area': ar, 'lgb': lr}, 'predictions': {'area': ar, 'lgb': lr},
                'new_native_fits': 2, 'new_NN_fits': 1, 'new_native_tree_fits': 1,
                'all_query_reload_exact': True, 'query_labels_available': False, 'query_scores_computed': False,
                'raw_price_pickle_opened': False, 'test_read': False, 'prior_or_reference_or_spatial_built': False,
                'preparation': r['preparation'], 'fit_label_manifest': r['fit_label_manifest'], 'request': c.record(request),
                'source_sha256': r['source_sha256'], 'input_sha256': r['input_sha256'], 'teachers': teachers,
                'geography': p.GEO_LIMIT, 'protected_before': r['protected_before'], 'protected_after': c.protected_hashes(),
                'fit_seconds': area_meta['fit_seconds'] + lgb_meta['fit_seconds'], 'elapsed_seconds': time.monotonic() - t}
        c.need(seal['protected_before'] == seal['protected_after'], 'Protected files changed')
        seal['artifact_sha256'] = {c.rel(f): c.sha(f) for f in sorted(out.rglob('*')) if f.is_file()}
        c.write(out / 'sealed_manifest.json', seal); status = 'completed'
        return seal
    except Exception:
        error = traceback.format_exc(); raise
    finally:
        c.write(out / 'execution.json', {'status': status, 'exit_code': 0 if status == 'completed' else 1,
                'failure': error, 'elapsed_seconds': time.monotonic() - t, 'native_fits_started': fits_started,
                'native_fits_completed': fits_completed, 'raw_price_pickle_opened': False, 'query_labels_read': False})


def self_check():
    checks = []
    def check(v, name): c.need(v, name); checks.append(name)
    cfg = expected_area_config()
    check(cfg['seed'] == 42 and cfg['epochs'] == 30 and cfg['batch_size'] == 2048, 'original area training schedule')
    params = lgb_parameters()
    check(params['n_estimators'] == 3999 and params['num_leaves'] == 63 and params['objective'] == 'regression_l1', 'original LGB 3999 unit recipe')
    X = pd.DataFrame({'building_area': [100., 20., np.nan], 'parking_area': [1., 2., 10.]})
    factor = light.price_factor(X, 'unit'); y = np.array([8e6, 3e6, 1e6], dtype=np.float64)
    check(np.array_equal(factor / 1e6, light.effective_area(X) / 100), 'original unit LGB total-MAE weights')
    check(np.array_equal((y / factor) * factor, y), 'unit target inverse exact on synthetic values')
    keys = set(p.PURE_ARRAYS) | {'train_y_million'}
    check(len(keys) == 8 and not any('query_y' in k for k in keys), 'exact R48 eight-key schema')
    tree = ast.parse(c.safe(__file__).read_text(encoding='utf-8-sig'))
    called = [n.func.attr for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)]
    check('read_pickle' not in called and 'load_data' not in called, 'teacher fit never opens raw pickle or generic data loader')
    check(not any(isinstance(n, ast.Import) and any(a.name == 'torch' for a in n.names) for n in ast.walk(tree)), 'Torch isolated subprocess only')
    check('fit_trees' not in called and 'train_nn' not in called and 'prepare_nn' not in called,
          'no high-level R90 multi-model preparation')
    return {'status': 'passed', 'check_count': len(checks), 'checks': checks, 'real_data_loaded': False,
            'new_native_fits': 0, 'plan_sha256': p.PLAN_SHA, 'source_sha256': p.source_hashes()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--freeze-request'); ap.add_argument('--preparation'); ap.add_argument('--labels')
    ap.add_argument('--run', action='store_true'); ap.add_argument('--request'); ap.add_argument('--out')
    ap.add_argument('--self-check', action='store_true'); ap.add_argument('--receipt')
    a = ap.parse_args()
    if a.self_check:
        t = time.monotonic(); result = self_check(); result['elapsed_seconds'] = time.monotonic() - t
        c.write(a.receipt, result); print(json.dumps(result, ensure_ascii=False))
    elif a.freeze_request: freeze(a.freeze_request, a.preparation, a.labels)
    elif a.run: run(a.request, a.out)
    else: ap.error('Choose freeze-request, run or self-check')
