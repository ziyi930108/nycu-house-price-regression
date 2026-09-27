"""R71 final saved-vector composition only: no fitting, NN prediction, or CSV writing."""
from pathlib import Path
import argparse, contextlib, hashlib, json, pickle, sys, time, traceback
import numpy as np
import pandas as pd
import round48_final_branch as final48

old, experiment = final48.old, final48.experiment
ROOT, spatial = final48.ROOT, final48.spatial
OUT = ROOT / 'outputs/round71/final'
PLAN = ROOT / 'outputs/round71/final_plan.json'
sha, rel, write = final48.sha, final48.rel, final48.write
FORMULA = 'median(spatial(reference(.75*(.5*saved_R54_area_native_TWD+.5*saved_A144_prior_native_TWD)+.25*old_unit_LGB_TWD)),R46_recency_log_spatial,R40_unit_spatial)'
EVIDENCE = {
 'outputs/round71/results.json': '8c9ddd281fb31df324b7aba4febd40c06cbee073ab45704b03b30176752ca833',
 'reports/round71_diagnostics.json': '3a9703780bea4750459069564e83f0eb6f58bebc6a5a0a3250b2492ec4169ba3',
 'outputs/round71/paired_vs_v18/results.json': '8e9c3e7dbf879afd0018fb517f829a6e675014ebf564d31c48a84c6a7ebd240d'}
AUTHORITIES = {
 'outputs/round55/final/trial_release.json': 'e3b3f4333a4d8aa056728819cb2853e8c296f5d2dd5f03e1a2f067e612badb0d',
 'outputs/round63/final/trial_release.json': '55e8fcffc1aef16dd056652191aff5363fa99332db97bcfd013956e88ce5b9a1',
 'outputs/round66/final/trial_release.json': '7fe4f2f027f027bb798e4c46aa2934a11b0e448331937869f09e0e7a70fe4a21'}


def run():
 started = time.monotonic(); inputs = {}
 def bind(name, digest):
  name = name.replace('\\', '/'); path = (ROOT / name).resolve()
  assert path.is_relative_to(ROOT.resolve()) and path.is_file(), name
  assert name not in inputs or inputs[name] == digest, 'Conflicting input: ' + name
  if name not in inputs: assert sha(path) == digest, 'Input changed: ' + name
  inputs[name] = digest
  return path
 def locked(name, digest):
  path = bind(name, digest); receipt = path.with_suffix('.sha256')
  assert receipt.read_text(encoding='ascii').strip() == digest, 'Receipt: ' + name
  bind(rel(receipt), sha(receipt))
  return json.loads(path.read_text(encoding='utf8'))
 plan = locked(rel(PLAN), sha(PLAN))
 assert plan['round'] == 71 and plan['proceed_final'] is True and plan['primary_candidate'] == 'R71'
 assert all(plan['locked_evidence'].get(n) == h for n, h in EVIDENCE.items()), 'Decision evidence differs'
 evidence = {n: locked(n, h) for n, h in plan['locked_evidence'].items()}
 history = evidence['outputs/round71/results.json']
 assert history['round'] == 71 and history['candidate'] == 'R71' and history['primary_seed'] == 42
 assert history['status'] == 'three_periods_completed_no_adoption_decision'
 assert history['checked_stages'] == list(experiment.STAGES) and history['formula'] == FORMULA
 assert history['new_NN_fits'] == history['model_predict_calls'] == 0 and not history['seed_average']
 p55, p63, p66 = [locked(n, h) for n, h in AUTHORITIES.items()]
 assert p63['status'] == 'experimental_v17_trial_generated' and p66['status'] == 'experimental_v18_trial_generated'
 assert all(p['fit_rows'] == 467221 and p['query_rows'] == p['reload_rows'] == 47073 and p['saved_model_reload_exact'] for p in (p55, p63, p66))
 assert p63['numeric_count'] == 144 and p63['seed'] == 42
 known, source = {}, {}
 for doc in (history, p55, p63, p66):
  for n, h in doc['source_sha256'].items():
   assert n not in source or source[n] == h, 'Conflicting source: ' + n
   source[n] = h
 for doc in (p55, p63, p66):
  for family in ('input_sha256', 'artifact_sha256'):
   for n, h in doc[family].items():
    n = n.replace('\\', '/')
    assert n not in known or known[n] == h, 'Conflicting authority: ' + n
    known[n] = h
 source[rel(Path(__file__))] = sha(Path(__file__))
 for n, h in source.items(): bind(n, h)
 assert source[rel(Path(final48.__file__))] == sha(Path(final48.__file__))
 def checked(name): return bind(name, known[name])
 def small(name): return json.loads(checked(name).read_text(encoding='utf8'))
 def vector(name, allow_zero=False):
  a = np.load(checked(name), allow_pickle=False)
  assert a.dtype == np.float64 and a.shape == (47073,) and np.isfinite(a).all() and ((a >= 0).all() if allow_zero else (a > 0).all()), name
  return a
 protected = dict(p66['protected_after']); protected[p66['path']] = p66['sha256']
 csv_names = {rel(p) for p in (ROOT / 'outputs').glob('submission_v*.csv')}
 assert len(csv_names) == 19 and csv_names == set(protected) - {'runs/versions.json'}, 'Old CSV inventory differs'
 for n, h in protected.items(): bind(n, h)
 # Canonical mapping is reproduced from target-free X only; no label cache is read.
 frames = {s: pd.read_pickle(checked(f'data_cache/{s}_features.pkl')) for s in ('train', 'valid', 'test')}
 schema = small('reports/feature_schema.json')
 for split, count in (('train', 455784), ('valid', 11994), ('test', 47073)):
  f = frames[split]
  assert len(f) == count and f.index.equals(pd.RangeIndex(count)) and list(f) == schema['columns']
  assert len(f.columns) == 87 and old.TARGET not in f
 def index(name, bound, count, identity=False):
  a = np.load(checked(name), allow_pickle=False)
  assert a.dtype == np.int64 and a.shape == (count,) and (np.diff(a) > 0).all() and a.min() >= 0 and a.max() < bound
  if identity: assert np.array_equal(a, np.arange(count, dtype='int64'))
  return a
 kt = index(old.GEO + 'train_source_cache_indices.npy', 455784, 455255)
 kv = index(old.GEO + 'valid_source_cache_indices.npy', 11994, 11994, True)
 kf = index(old.GEO + 'full_source_cache_indices.npy', 467249, 467221)
 kq = index(old.GEO + 'test_source_cache_indices.npy', 47073, 47073, True)
 X = pd.concat([frames['train'].iloc[kt], frames['valid'].iloc[kv]], ignore_index=True).iloc[kf].reset_index(drop=True)
 Xq = frames['test'].iloc[kq].reset_index(drop=True); del frames
 geo, geoq = [pd.read_pickle(checked(old.GEO + n)) for n in ('full.pkl', 'test.pkl')]
 before = {n: spatial.frame_sha(f) for n, f in (('X', X), ('Xq', Xq), ('geo', geo), ('geoq', geoq))}
 assert before == p55['input_frame_sha256'] == p63['input_frame_sha256'] == p66['input_frame_sha256']
 target_identity = p63['fit_target_values_sha256']
 assert target_identity == p55['fit_target_values_sha256'] == p66['fit_target_values_sha256']
 for kind, rows in (('fit', 467221), ('query', 47073)):
  indices = [index(f'outputs/round{r}/final/{kind}_indices.npy', rows, rows, True) for r in (55, 63, 66)]
  assert all(np.array_equal(indices[0], a) for a in indices[1:])
  final48.array(OUT / f'{kind}_indices.npy', indices[0])
 # Model/checkpoint bytes and saved prediction metadata are identified, never loaded as models.
 native_metadata = {}
 for r, width, trainer in ((55, 603, 'src/round48_nn_train.py'), (63, 144, 'src/round63_recency_weighted_prior_train.py')):
  prefix = f'outputs/round{r}/final/model/'
  checked(prefix + 'model.pt'); m = small(prefix + 'training.json')
  assert m['status'] == 'completed' and m['source_sha256'] == source[trainer]
  assert m['fit_rows'] == 467221 and m['query_rows'] == m['reload_rows'] == 47073
  assert m['numeric_count'] == width and m['configuration']['seed'] == 42 and m['configuration']['epochs'] == 30
  assert m['prediction_units'] == 'TWD' and m['saved_model_reload_exact'] and not m['query_labels_available']
  assert m['input_sha256'] == known[f'outputs/round{r}/final/prepared.npz']
  native_metadata[str(r)] = m
 assert native_metadata['63']['configuration'] == p63['training_configuration']
 assert native_metadata['63']['train_sample_weight_sha256'] == p63['train_sample_weight_sha256']
 area = vector('outputs/round55/final/model/predictions_twd.npy')
 prior = vector('outputs/round63/final/model/predictions_twd.npy')
 recency, unit = vector(final48.RECENCY), vector(final48.inherited.UNIT_BRANCH)
 oldunit = vector(old.OLD_UNIT)
 mean = .5 * area + .5 * prior
 controls = {'v17': vector('outputs/round63/final/predictions.npy'), 'v18': vector('outputs/round66/final/predictions.npy')}
 checks = {'v17_native_mean': final48.exact(mean, vector('outputs/round63/final/native_mean.npy'), 'v17 native mean'),
  'v17_base': final48.exact(.75 * (.5 * vector(old.OLD_RAW, allow_zero=True) + .5 * mean) + .25 * oldunit, vector('outputs/round63/final/base.npy'), 'v17 base'),
  'v18_base': final48.exact(.75 * vector('outputs/round66/final/native_mean.npy') + .25 * oldunit, vector('outputs/round66/final/base.npy'), 'v18 base')}
 for label, r in (('v17', 63), ('v18', 66)):
  checks[label] = final48.exact(np.median(np.column_stack([vector(f'outputs/round{r}/final/nn_spatial.npy'), recency, unit]), axis=1), controls[label], label)
 sample_path = rel(old.SAMPLE); sample = pd.read_csv(checked(sample_path), encoding='utf-8-sig', usecols=[old.ID], dtype={old.ID: str}, keep_default_na=False)
 ids = sample[old.ID].tolist()
 assert len(ids) == 47073 and sample[old.ID].is_unique and all(isinstance(v, str) and v for v in ids)
 old_prices = {name: old.rounded(values) for name, values in controls.items()}
 for name in controls: final48.csv_check(ROOT / f'outputs/submission_{name}_trial.csv', ids, old_prices[name])
 manifest = small(final48.inherited.GEO_MANIFEST)
 newquery = pd.read_pickle(checked(manifest['features']['query']['path']))
 assert spatial.frame_sha(newquery) == manifest['features']['query']['frame_sha256']
 assert newquery.index.equals(Xq.index) and newquery.dtypes.equals(geoq.dtypes) and list(newquery) == list(geoq)
 with checked(old.REFERENCE).open('rb') as f: reference = pickle.load(f)
 with checked(final48.FINAL + 'spatial_state.pkl').open('rb') as f: state = pickle.load(f)
 summary = small(final48.FINAL + 'state_summary.json')
 assert state['state_summary'] == summary and summary['fit_rows'] == len(X) == 467221
 assert not summary['query_used_in_build'] and summary['fit_X_sha256'] == before['X']
 assert summary['fit_geo_sha256'] == manifest['features']['fit']['frame_sha256'] and summary['parameters'] == spatial.parameters()
 assert reference['fit_report']['fitting_rows'] == 467221
 assert summary['existing_reference_residual_sha256'] == experiment.original.value_sha(reference['reference_residual'], '<f8')
 write(OUT / 'input_contract.json', {'round': 71, 'source_sha256': source, 'input_sha256': inputs,
  'final_plan_sha256': inputs[rel(PLAN)], 'formula': FORMULA, 'protected_before': protected,
  'prepared_NPZ_identity_inherited_not_loaded': {str(r): known[f'outputs/round{r}/final/prepared.npz'] for r in (55, 63)},
  'fit_target_values_sha256_inherited': target_identity, 'fit_target_values_rehashed_this_run': False})
 B = .75 * mean + .25 * oldunit
 p, diag, rr = experiment.original.predict_reference(reference, Xq, B, variant='trend', return_diagnostics=True)
 branch, detail, sr = spatial.predict(state, Xq, newquery, B, p, diag)
 fallback = ~detail['applied'].to_numpy(dtype=bool)
 assert branch[fallback].tobytes() == p[fallback].tobytes(), 'Spatial fallback differs'
 candidate = np.median(np.column_stack([branch, recency, unit]), axis=1)
 for name, a in (('native_mean', mean), ('base', B), ('reference', p), ('nn_spatial', branch), ('predictions', candidate)):
  assert a.dtype == np.float64 and a.shape == (47073,) and np.isfinite(a).all() and (a > 0).all(), name
  final48.array(OUT / (name + '.npy'), a)
 for name, obj in (('reference_diagnostics', diag), ('spatial_diagnostics', detail)):
  with (OUT / (name + '.pkl')).open('xb') as f: pickle.dump(obj, f, protocol=5)
 final48.exact(candidate, np.load(OUT / 'predictions.npy', allow_pickle=False), 'Saved final candidate')
 prices = old.rounded(candidate)  # The candidate is rounded exactly once; CSV authors consume these integers.
 assert prices.dtype == np.int64 and prices.shape == (47073,) and (prices > 0).all()
 write(OUT / 'export_payload.json', {'headers': [old.ID, old.TARGET], 'ids': ids, 'prices': prices.tolist(),
  'round': 71, 'rows': 47073, 'prediction_path': rel(OUT / 'predictions.npy'), 'prediction_sha256': sha(OUT / 'predictions.npy'),
  'rounding': 'maximum(rint(unrounded candidate TWD),1).astype(int64), once',
  'sample_ID_path': sample_path, 'sample_ID_sha256': inputs[sample_path]})
 differences = {}
 for label, baseline in controls.items():
  delta = candidate - baseline
  differences[label] = {'changed_unrounded_rows': int(np.count_nonzero(delta)),
   'changed_rounded_rows': int(np.count_nonzero(prices != old_prices[label])),
   'mean_absolute_prediction_change_TWD': float(np.abs(delta).mean()),
   'maximum_absolute_prediction_change_TWD': float(np.abs(delta).max())}
 assert before == {n: spatial.frame_sha(f) for n, f in (('X', X), ('Xq', Xq), ('geo', geo), ('geoq', geoq))}
 assert spatial.frame_sha(newquery) == manifest['features']['query']['frame_sha256']
 assert all(sha(ROOT / n) == h for n, h in source.items()), 'Source changed'
 assert sha(PLAN) == inputs[rel(PLAN)] and all(sha(ROOT / n) == h for n, h in EVIDENCE.items())
 after = {n: sha(ROOT / n) for n in protected}
 assert after == protected and csv_names == {rel(f) for f in (ROOT / 'outputs').glob('submission_v*.csv')}
 metadata = {'round': 71, 'formula': FORMULA, 'fit_rows': 467221, 'query_rows': 47073,
  'input_frame_sha256': before, 'fit_target_values_sha256_inherited': target_identity, 'targets_loaded': False,
  'reused_native_training_metadata': native_metadata, 'reused_prior_numeric_count': 144, 'seed': 42,
  'source_sha256': source, 'input_sha256': inputs, 'controls': checks, 'old_v17_v18_CSV_exact': True,
  'reference_report': rr, 'spatial_report': sr, 'fallback_bitwise_equal': True,
  'prediction_units': 'TWD', 'prediction_min_TWD': float(candidate.min()), 'prediction_max_TWD': float(candidate.max()),
  'prediction_changes_vs_controls': differences, 'prediction_changes_are_not_errors': True,
  'new_NN_fits': 0, 'new_tree_fits': 0, 'model_predict_calls': 0, 'models_loaded': 0,
  'reference_predict_calls': 1, 'spatial_predict_calls': 1, 'new_spatial_state_builds': 0,
  'encoder_prior_weights_refit': False, 'test_labels_loaded': False, 'sample_ID_order_verified': True,
  'protected_before': protected, 'protected_after': after, 'old_CSV_unchanged': True,
  'new_CSV': False, 'registry_updated': False, 'selector_updated': False, 'uploaded': False}
 write(OUT / 'metadata.json', metadata)
 proof = dict(metadata, status='final_vectors_and_export_payload_ready_no_CSV',
  final_plan_sha256=inputs[rel(PLAN)], locked_evidence=plan['locked_evidence'],
  prediction_path=rel(OUT / 'predictions.npy'), prediction_sha256=sha(OUT / 'predictions.npy'),
  export_payload_path=rel(OUT / 'export_payload.json'), export_payload_sha256=sha(OUT / 'export_payload.json'),
  sample_ID_path=sample_path, sample_ID_sha256=inputs[sample_path], positive_int64_payload=True,
  rounding='maximum(rint(unrounded candidate TWD),1).astype(int64), once',
  public_mae=None, private_mae=None, elapsed_seconds=time.monotonic() - started, completed_at_utc=old.stamp(),
  artifact_sha256={rel(f): sha(f) for f in sorted(OUT.rglob('*')) if f.is_file() and f.name != 'execution.log'})
 write(OUT / 'trial_release.json', proof)
 print(json.dumps({'status': proof['status'], 'new_NN_fits': 0, 'model_predict_calls': 0,
  'rows': 47073, 'differences': differences, 'elapsed_seconds': proof['elapsed_seconds']}, ensure_ascii=False), flush=True)


def main():
 parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true', required=True)
 parser.parse_args(); assert not OUT.exists(), 'Existing final attempt; no overwrite or automatic retry'
 OUT.mkdir(parents=True, exist_ok=False); started = old.stamp(); tick = time.monotonic(); code = 1; failure = None
 write(OUT / 'started.json', {'round': 71, 'started_at_utc': started, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
 with (OUT / 'execution.log').open('x', encoding='utf8') as stream:
  with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
   try: run(); code = 0
   except Exception: failure = traceback.format_exc(); print(failure, flush=True)
 event = {'round': 71, 'command': sys.argv, 'started_at_utc': started, 'completed_at_utc': old.stamp(),
  'elapsed_seconds': time.monotonic() - tick, 'exit_code': code, 'failure': failure, 'automatic_retry': False,
  'source_sha256': sha(Path(__file__)), 'log_path': rel(OUT / 'execution.log'), 'log_sha256': sha(OUT / 'execution.log'),
  'new_NN_fits': 0, 'model_predict_calls': 0, 'new_CSV': False}
 write(OUT / 'execution.json', event); print(json.dumps(event, ensure_ascii=False), flush=True)
 return code


if __name__ == '__main__': raise SystemExit(main())
