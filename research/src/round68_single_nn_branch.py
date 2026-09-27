"""R68: use saved R66 corrected NN mixture directly; remove only its outer median."""
import argparse
import json
from pathlib import Path
import sys
import time
import traceback
import numpy as np
import round66_nn_branch_ablation as parent

old, ROOT, STAGES = parent.old, parent.ROOT, parent.STAGES
OUT, DOCUMENT = ROOT/'outputs/round68', ROOT/'reports/round68_plan.md'
sha, read, write, rel = old.sha, old.locked, old.write, old.rel
FORMULA = 'saved_R66_nn_spatial_TWD (outer recency_log/unit median removed; no new reference/spatial calls)'
CONTROLS = ('v18', 'v17', 'R65', 'v16', 'v15', 'v13', 'v11', 'v12', 'v5', 'R62')
ZERO = dict(new_NN_fits=0, model_predict_calls=0, reference_predict_calls=0, spatial_predict_calls=0)


def run():
    started = time.monotonic(); inputs = {}
    def checked(name, digest):
        name = name.replace('\\', '/'); path = ROOT/name
        assert name not in inputs or inputs[name] == digest, name
        if name not in inputs: assert sha(path) == digest, name
        inputs[name] = digest; return path
    def small(name):
        path = ROOT/name; value = read(path)
        inputs[name] = sha(path); inputs[rel(path.with_suffix('.sha256'))] = sha(path.with_suffix('.sha256'))
        return value
    p48, p52, prior = (small(p) for p in ('outputs/round48/plan.json', 'outputs/round52/plan.json', 'outputs/round66/results.json'))
    assert prior['round'] == 66 and prior['candidate'] == 'R66' and prior['checked_stages'] == list(STAGES)
    assert prior['status'] == 'three_periods_completed_no_adoption_decision' and prior['formula'] == parent.FORMULA
    source = dict(prior['source_sha256']); source[rel(Path(__file__))] = sha(Path(__file__))
    assert all(sha(ROOT/n) == h for n, h in source.items()), 'Inherited source changed'
    document_sha = sha(DOCUMENT); inputs[rel(DOCUMENT)] = document_sha
    protected = {rel(p): sha(p) for p in sorted((ROOT/'outputs').glob('submission_v*.csv'))}
    protected['runs/versions.json'] = sha(ROOT/'runs/versions.json')
    data = {n: p48['input_sha256'][n] for n in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json', old.previous.KT)}
    for n in ('reports/feature_schema.json', old.previous.KT): checked(n, data[n])
    write(OUT/'plan.json', dict(round=68, candidate='R68', primary_control='v18=R66', formula=FORMULA,
        underlying_base_formula=parent.BASE_FORMULA, native_mean_formula=parent.NATIVE_FORMULA,
        source_sha256=source, text_plan_sha256=document_sha, stage_order=list(STAGES),
        inherited_data_sha256=data, data_cache_hashes_inherited_not_recomputed=True,
        split_contract=p48['split_contract'], three_periods_regardless_of_scores=True, new_gate=None, **ZERO))
    Xall, yall = old.previous.load_training_rows(p48)
    result = dict(round=68, candidate='R68', status='running', formula=FORMULA, primary_control='v18',
        base_formula=parent.BASE_FORMULA, native_mean_formula=parent.NATIVE_FORMULA, source_sha256=source,
        plan_sha256=document_sha, machine_plan_sha256=sha(OUT/'plan.json'), checked_stages=[], stages={},
        protected_before=protected, new_submission_generated=False, fresh_blind_validation=False,
        inherited_data_sha256=data, data_cache_hashes_inherited_not_recomputed=True, **ZERO)
    for stage in STAGES:
        tick = time.monotonic(); a = prior['stages'][stage]; folder = OUT/stage; folder.mkdir()
        artifacts = {n.replace('\\', '/'): h for n, h in a['artifact_sha256'].items()}
        def saved(name): return checked(name, artifacts[name])
        fi, qi = [np.load(saved(f'outputs/round66/{stage}/{kind}_indices.npy'), allow_pickle=False) for kind in ('fit', 'query')]
        for kind, ind in (('fit', fi), ('query', qi)):
            c = p48['split_contract'][stage]
            assert ind.dtype == np.int64 and ind.shape == (c[kind+'_rows'],) and (np.diff(ind) > 0).all()
            assert ind.min() >= 0 and ind.max() < len(Xall) and old.original.value_sha(ind, '<i8') == c[kind+'_indices_content_sha256']
            old.previous.save_array(folder/f'{kind}_indices.npy', ind)
        assert not np.intersect1d(fi, qi).size and a['dates'] == p48['splits'][stage]
        X, Xq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True)
        frames = {'fit_X': old.spatial.frame_sha(X), 'query_X': old.spatial.frame_sha(Xq)}
        assert frames == a['original_frame_sha256'], 'Canonical frames changed'
        candidate_path = f'outputs/round66/{stage}/nn_spatial.npy'
        candidate = old.vector(saved(candidate_path), len(qi))
        vectors = {'v18': old.vector(saved(f'outputs/round66/{stage}/predictions.npy'), len(qi))}
        names = {label: f'outputs/round{r}/{stage}/predictions.npy' for label, r, _ in parent.CONTROLS}
        names['v5'] = p48['bindings'][stage]['v5']
        for label, name in names.items(): vectors[label] = old.vector(checked(name, prior['input_sha256'][name]), len(qi))
        log_name, unit_name = p52['bindings'][stage]['recency_log'], p48['bindings'][stage]['unit_spatial']
        log = old.vector(checked(log_name, prior['input_sha256'][log_name]), len(qi))
        unit = old.vector(checked(unit_name, prior['input_sha256'][unit_name]), len(qi))
        old.exact(vectors['v18'], np.median(np.column_stack([candidate, log, unit]), axis=1), 'R66 outer median control')
        score = lambda v: old.original.score(yall[qi], v, X, Xq, p48['splits'][stage][0], p48['target_group_weights'])
        metrics = {label: score(vectors[label]) for label in CONTROLS}
        assert all(metrics[label] == a['metrics']['R66' if label == 'v18' else label] for label in CONTROLS), 'Full control metrics changed'
        metrics['R68'] = score(candidate); old.previous.save_array(folder/'predictions.npy', candidate)
        old.exact(candidate, old.vector(folder/'predictions.npy', len(qi)), 'Saved candidate unchanged')
        item = dict(round=68, stage=stage, status='completed', fit_rows=len(fi), query_rows=len(qi), dates=p48['splits'][stage],
            metrics=metrics, differences={label: old.previous.prior.log_run.differences(metrics[label], metrics['R68']) for label in CONTROLS},
            original_frame_sha256=frames, controls_full_metrics_exact=True, R66_median_control_bitwise_exact=True,
            saved_candidate_bitwise_exact=True, candidate_source_path=candidate_path, candidate_source_sha256=artifacts[candidate_path],
            query_labels_used_only_by_scorer=True, prediction_sha256=sha(folder/'predictions.npy'),
            artifact_sha256={rel(p): sha(p) for p in folder.iterdir() if p.is_file()}, elapsed_seconds=time.monotonic()-tick, **ZERO)
        write(folder/'results.json', item); result['stages'][stage] = item; result['checked_stages'].append(stage)
        print(json.dumps({'stage': stage, 'vs_v18': item['differences']['v18'], **ZERO}), flush=True)
    assert all(sha(ROOT/n) == h for n, h in source.items()) and sha(DOCUMENT) == document_sha
    result.update(status='three_periods_completed_no_adoption_decision', input_sha256=inputs,
        protected_after={n: sha(ROOT/n) for n in protected}, elapsed_seconds=time.monotonic()-started)
    assert result['protected_after'] == protected
    write(OUT/'results.json', result); print(json.dumps({'status': result['status'], 'elapsed_seconds': result['elapsed_seconds']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--run', action='store_true'); args = parser.parse_args()
    if not args.run: parser.error('--run required')
    assert not OUT.exists(), 'Existing output: no overwrite'
    OUT.mkdir(); tick = time.monotonic(); started = old.now(); code, failure = 0, None
    write(OUT/'started.json', {'started_at_utc': started, 'command': sys.argv, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    try: run()
    except Exception:
        code, failure = 1, traceback.format_exc(); print(failure, flush=True)
    write(OUT/'execution.json', {'exit_code': code, 'failure': failure, 'started_at_utc': started, 'completed_at_utc': old.now(),
        'elapsed_seconds': time.monotonic()-tick, 'source_sha256': sha(Path(__file__)), 'automatic_retry': False})
    raise SystemExit(code)
