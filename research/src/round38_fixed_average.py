"""One fixed 50:50 mean of two completed R36 prediction methods; no models."""
from pathlib import Path
import argparse
import time
import traceback
import numpy as np
import pandas as pd
import round36_combination as prior

ROOT = prior.ROOT
OUT = ROOT / 'outputs/round38'
DOC = ROOT / 'reports/round38_fixed_average_plan.md'
PLAN36 = ROOT / 'outputs/round36/combination_plan.json'
RESULT36 = ROOT / 'outputs/round36/results.json'
PLAN34 = ROOT / 'outputs/round34/trial_plan.json'
KT = 'outputs/round6/geography/final/train_source_cache_indices.npy'
METHODS = (prior.V3, prior.SPATIAL, prior.COMBINED)
CANDIDATE = 'fixed_half_spatial_half_log_spatial'
need, sha, rel, locked, write = prior.require, prior.sha, prior.rel, prior.locked, prior.locked_write
PROTECTED = ('outputs/submission_v2.csv', 'outputs/submission_v3.csv',
             'outputs/submission_v4_trial.csv', 'runs/versions.json',
             'outputs/round36/combination_plan.json', 'outputs/round36/results.json')


def sources():
    return prior.merge(prior.sources(), {rel(Path(__file__)): sha(Path(__file__))})


def runtime():
    return prior.spatial_run.original.runtime_versions()


def freeze():
    need(not (OUT / 'plan.json').exists() and not (OUT / 'run_claim.json').exists(), 'Existing attempt')
    need(sha(PLAN36) == '26194404d158613f0837e49c4cdb97b4c9c1ef213b1561cfd7ac80dddc4fc642', 'R36 plan changed')
    need(sha(RESULT36) == '133d99e9fdca21ddf97e58fa8c7860ac74f7311bfe56f0005244c3a76791ef9e', 'R36 result changed')
    need(sha(PLAN34) == 'aca3e5cb5f606a8cd2603efd4f56634cd8ef388fe760614f3b0b745f32b2add9', 'R34 plan changed')
    p36, r36, p34 = locked(PLAN36), locked(RESULT36), locked(PLAN34)
    need(r36['status'] == 'three_periods_completed_no_adoption_decision'
         and r36['checked_stages'] == list(prior.STAGES)
         and prior.sources() == p36['source_sha256'], 'Prior research incomplete or changed')
    inputs = {rel(p): sha(p) for p in (PLAN36, RESULT36, PLAN34, DOC)}
    for name in ('data_cache/train.pkl', 'data_cache/train_features.pkl', 'reports/feature_schema.json'):
        inputs[name] = p36['dependency_sha256'][name]
    inputs[KT] = p34['input_sha256'][KT]
    for stage in prior.STAGES:
        path = RESULT36.parent / stage / 'results.json'
        need(sha(path) == r36['stage_result_sha256'][stage], 'Prior stage result changed')
        result = locked(path)
        inputs[rel(path)] = sha(path)
        for name in ('fit_indices', 'query_indices', *METHODS):
            key = rel(path.parent / (name + '.npy'))
            inputs[key] = result['artifact_sha256'][key]
    write(OUT / 'plan.json', {'round': 38, 'created_at_utc': prior.now(),
          'formula': '0.5*spatial_v3+0.5*log_spatial_combination',
          'stage_order': list(prior.STAGES), 'source_sha256': sources(), 'runtime': runtime(),
          'input_sha256': inputs, 'protected_sha256': {n: sha(ROOT / n) for n in PROTECTED},
          'target_group_weights': p36['target_group_weights'], 'splits': p36['splits'],
          'split_contract': p36['split_contract'], 'fresh_blind_validation': False,
          'new_models_trained': 0, 'tree_models_loaded': 0, 'submission_generated': False,
          'formal_adoption_decision': None})
    print('fixed_average_plan_frozen', sha(OUT / 'plan.json'), flush=True)


def run():
    p = locked(OUT / 'plan.json')
    need(p['source_sha256'] == sources() and p['runtime'] == runtime(), 'Source/runtime changed')
    need(not (OUT / 'run_claim.json').exists() and not (OUT / 'results.json').exists(), 'Existing attempt')
    write(OUT / 'run_claim.json', {'started_at_utc': prior.now(), 'plan_sha256': sha(OUT / 'plan.json')})
    result = {'round': 38, 'status': 'incomplete', 'plan_sha256': sha(OUT / 'plan.json'),
              'stages': {}, 'checked_stages': [], 'new_models_trained': 0,
              'tree_models_loaded': 0, 'submission_generated': False, 'formal_adoption_decision': None}
    start = time.monotonic()
    try:
        prior.c6.verify_hashes(prior.merge(p['input_sha256'], p['source_sha256'], p['protected_sha256']))
        Xraw = pd.read_pickle(ROOT / 'data_cache/train_features.pkl')
        raw = pd.read_pickle(ROOT / 'data_cache/train.pkl')
        kt = np.load(ROOT / KT, allow_pickle=False)
        schema = prior.c6.read(ROOT / 'reports/feature_schema.json')
        need(len(Xraw) == len(raw) == 455784 and list(Xraw.columns) == schema['columns']
             and Xraw.index.equals(pd.RangeIndex(455784)) and raw.index.equals(Xraw.index), 'Raw X/y schema differs')
        need(kt.dtype == np.dtype('int64') and kt.shape == (455255,)
             and (np.diff(kt) > 0).all() and kt.min() >= 0 and kt.max() < len(raw), 'Canonical indices differ')
        Xall = Xraw.iloc[kt].reset_index(drop=True)
        yall = raw['總價元'].to_numpy(dtype='float64')[kt]
        del Xraw, raw
        for stage in p['stage_order']:
            folder = RESULT36.parent / stage
            old = locked(folder / 'results.json')
            fi, qi = [np.load(folder / (n + '_indices.npy'), allow_pickle=False) for n in ('fit', 'query')]
            for name, values in (('fit', fi), ('query', qi)):
                spec = p['split_contract'][stage]
                need(values.dtype == np.dtype('int64') and len(values) == spec[name + '_rows']
                     and (np.diff(values) > 0).all() and values.min() >= 0 and values.max() < len(Xall), 'Period indices differ')
                need(prior.log_run.original.value_sha(values, '<i8') == spec[name + '_indices_content_sha256'], 'Period index identity differs')
            need(not np.intersect1d(fi, qi).size, 'Fit/query overlap')
            X, Xq, yq = Xall.iloc[fi].reset_index(drop=True), Xall.iloc[qi].reset_index(drop=True), yall[qi]
            arrays = {name: np.load(folder / (name + '.npy'), allow_pickle=False) for name in METHODS}
            for values in arrays.values():
                need(values.dtype == np.dtype('float64') and values.shape == (len(qi),)
                     and np.isfinite(values).all() and (values >= 0).all(), 'Invalid prior prediction')
            score = lambda values: prior.log_run.original.score(yq, values, X, Xq, p['splits'][stage][0], p['target_group_weights'])
            metrics = {name: score(values) for name, values in arrays.items()}
            need(all(metrics[name] == old['metrics'][name] for name in METHODS), 'Prior full metric reproduction failed')
            combined = .5 * arrays[prior.SPATIAL] + .5 * arrays[prior.COMBINED]
            need(np.isfinite(combined).all() and (combined >= 0).all(), 'Invalid average')
            metrics[CANDIDATE] = score(combined)
            dest = OUT / stage
            dest.mkdir(exist_ok=False)
            prior.save_array(dest / 'predictions.npy', combined)
            back = np.load(dest / 'predictions.npy', allow_pickle=False)
            need(back.dtype == combined.dtype and back.tobytes() == combined.tobytes(), 'Saved vector changed')
            item = {'stage': stage, 'metrics': metrics,
                    'differences': {name: prior.log_run.differences(metrics[name], metrics[CANDIDATE]) for name in METHODS},
                    'prior_full_metrics_reproduced': True, 'saved_prediction_bitwise_equal': True,
                    'prediction_sha256': sha(dest / 'predictions.npy'), 'fit_rows': len(fi), 'query_rows': len(qi)}
            write(dest / 'results.json', item)
            result['stages'][stage] = item
            result['checked_stages'].append(stage)
            print(stage, item['differences'][prior.V3], flush=True)
        prior.c6.verify_hashes(prior.merge(p['protected_sha256'], p['source_sha256']))
        result['protected_after'] = {name: sha(ROOT / name) for name in PROTECTED}
        result['status'] = 'three_periods_completed_no_adoption_decision'
    except Exception as exc:
        result.update(status='incomplete_technical_failure', error=str(exc), traceback=traceback.format_exc())
    result.update(completed_at_utc=prior.now(), elapsed_seconds=time.monotonic() - start)
    write(OUT / 'results.json', result)
    print(result['status'], sha(OUT / 'results.json'), flush=True)
    return 0 if result['status'] == 'three_periods_completed_no_adoption_decision' else 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--freeze', action='store_true')
    group.add_argument('--run', action='store_true')
    args = parser.parse_args()
    if args.freeze:
        freeze()
    else:
        raise SystemExit(run())
