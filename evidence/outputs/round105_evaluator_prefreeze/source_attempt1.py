"""R105 isolated zero-fit scorer; both endpoint replays seal before query labels.

C is the saved complete R102T endpoint, not a newly trained mechanism control.
Only the mature R100 metric/group/bootstrap functions are reused; authority and
prelabel checks below belong to the fixed R105 native-average experiment.
"""
from __future__ import annotations
import argparse
import copy
import json
import time
import traceback
import numpy as np
import pandas as pd
import round100_contract as c
import round100_temporal_prior_evaluate_v2 as metric

PLAN = 'reports/round105_plan.md'
PLAN_SHA = '8eed243d95b7c76c583b42e17b10d95aaaefc166245aaafd2c9af11dffb90643'
OLD = {'path': 'outputs/round102/sealed_manifest.json', 'sha256': 'fd75ffd27998d008cedb65a13632e48c480f05beb9c69ee9bccf5b2e7a1aa68a'}
OLD_RESULTS = {'path': 'outputs/round102/evaluation/results.json', 'sha256': 'e385ed630cdf3a2b2b0418867563a4064fbe08c1c979feb004787df2592f2347'}
LAYERS = ('native', 'base', 'reference', 'spatial', 'median')
ZERO_COUNTS = ('new_native_fits', 'new_NN_fits', 'new_native_tree_fits', 'new_fit_calls', 'prior_or_R_S_build_calls')
REPLAY_FLAGS = ('original_v22_all_five_layers_replay_exact', 'original_R102T_all_five_layers_replay_exact',
                'all_endpoints_replay_exact', 'R_S_state_and_frames_unchanged')
CAUTION = ('Adaptive equal-weight hypothesis on reused windows, not a new blind test or measured variance reduction. '
           'C is the saved complete R102T endpoint, not a retrained or causal mechanism control. '
           'Static 2023 geography is not fully historically as-of. Conditional bootstrap is not Private probability. '
           'Only v22 remains selected; no alpha search or automatic adoption.')


def parameters():
    p = c.evaluation_parameters()
    p.pop('point_only_control'); p.pop('mechanism_control')
    p.update(round=105, endpoint_control='C', endpoint_control_source='saved complete R102T',
             candidate_role='fixed native .5*v22_prior_NN + .5*R102T_prior_NN',
             alpha=.5, supplementary_endpoint_required_metrics=['overall_mae', 'weighted_mae'],
             supplementary_endpoint_required_layer='median', native_improvement_required=False,
             variance_reduction_measured=False, new_fits=0, fully_historically_as_of=False)
    return p


def source_hashes():
    return c.source_hashes(__file__)


def research_gates(stages):
    primary = metric.research_gate(stages)
    primary.pop('C_is_mechanism_control_not_alternative_candidate')
    parts = {'all_three_final_' + k + '_improve': all(
        stages[s]['metrics']['T'][k] < stages[s]['metrics']['C'][k] for s in c.STAGES)
        for k in ('overall_mae', 'weighted_mae')}
    supplementary = {'components': parts, 'supplementary_endpoint_gate_passed': all(parts.values()),
                     'control': 'saved complete R102T', 'native_improvement_required': False,
                     'causal_identification_claimed': False}
    return {'primary': primary, 'supplementary_endpoint': supplementary,
            'both_research_gates_passed': primary['adoption_research_gate_passed'] and all(parts.values()),
            'automatic_adoption': False, 'selected_submission': 'v22', 'Private_score': None, 'new_CSV': False}


def check_header(seal):
    c.need(seal['schema_version'] == 1 and seal['status'] == 'all_native_average_predictions_sealed' and
           seal['stage_order'] == list(c.STAGES) and set(seal['stages']) == set(c.STAGES), 'Three complete stages required')
    c.need(seal['plan_sha256'] == PLAN_SHA and all(seal[k] == 0 for k in ZERO_COUNTS), 'Plan or zero-fit counts differ')
    c.need(seal['all_query_reload_exact'] is True and all(seal[k] is True for k in REPLAY_FLAGS), 'Endpoint replay/reload incomplete')
    c.need(all(seal[k] is False for k in ('query_labels_available', 'query_scores_computed', 'raw_price_pickle_opened')), 'Prediction label boundary differs')
    c.need(len(seal['protected_before']) == 9 and seal['protected_before'] == seal['protected_after'], 'Protected identities differ')


def check_native_mean(layers):
    c.exact(.5 * layers['v22']['native'] + .5 * layers['C']['native'], layers['T']['native'], 'Fixed native mean differs')


def preflight(sealed_path, request_path):
    checked = {}
    def bind(path, digest):
        path = c.rel(path)
        c.need(path not in checked or checked[path] == digest, 'Conflicting file identity')
        c.need(c.sha(path) == digest, 'Input SHA differs: ' + path); checked[path] = digest
        return c.safe(path)
    def document(rec): return c.read(bind(rec['path'], rec['sha256']))
    def array(rec, semantic=True):
        a = np.load(bind(rec['path'], rec['sha256']), allow_pickle=False)
        c.need(isinstance(a, np.ndarray) and not a.dtype.hasobject, 'Numeric array required')
        if semantic: c.need(c.identity(a) == {k: rec[k] for k in ('dtype', 'shape', 'values_sha256')}, 'Array identity differs')
        return a
    request = document(c.record(request_path))
    c.need(request['schema_version'] == 1 and request['plan'] == {'path': PLAN, 'sha256': PLAN_SHA} and
           request['parameters'] == parameters(), 'Evaluation request differs')
    bind(PLAN, PLAN_SHA); own = source_hashes()
    c.need(all(request['source_sha256'].get(k) == v for k, v in own.items()), 'Evaluator sources not frozen')
    c.verify_hashes(request['source_sha256'])
    launch = document(request['launch_request'])
    c.need(launch['plan_sha256'] == PLAN_SHA, 'Research request plan differs'); c.verify_hashes(launch['source_sha256'])
    c.need(request['protected_sha256'] == c.protected_hashes(), 'Protected files changed')
    c.need(c.rel(sealed_path) == 'outputs/round105/sealed_manifest.json', 'Unexpected root seal path')
    seal = document(c.record(sealed_path)); check_header(seal)
    c.need(seal['request'] == request['launch_request'] and seal['protected_after'] == request['protected_sha256'], 'Root binding differs')
    execution = c.safe(sealed_path).parent / 'execution.json'
    ex = document({'path': c.rel(execution), 'sha256': execution.with_suffix('.sha256').read_text(encoding='ascii').strip()})
    c.need(ex['exit_code'] == 0 and ex.get('failure') is None, 'Root process incomplete')
    c.verify_hashes(seal['source_sha256'])
    c.need(all(launch['source_sha256'].get(k) == v for k, v in seal['source_sha256'].items()), 'Driver source not frozen')
    oldmain = document(OLD); oldresults = document(OLD_RESULTS)
    for k, h in own.items():
        if k in oldresults['source_sha256']: c.need(h == oldresults['source_sha256'][k], 'Reused metric source drift')
    for key in ('features', 'canonical_mapping', 'raw_train'):
        c.need(seal[key] == oldmain[key], 'Dataset identity changed'); bind(seal[key]['path'], seal[key]['sha256'])
    mapping = array(seal['canonical_mapping'], False)
    metric.index(mapping, 455784, 'canonical mapping'); c.need(len(mapping) == 455255, 'Canonical row count differs')
    Xraw = pd.read_pickle(c.safe(c.FEATURES))
    c.need(len(Xraw) == 455784 and Xraw.index.equals(pd.RangeIndex(len(Xraw))) and
           not {'總價元', 'target', 'y'}.intersection(Xraw.columns), 'Pure-X schema differs')
    days = Xraw.iloc[mapping].trade_day_index.to_numpy(dtype=np.float64); stages = {}
    for s in c.STAGES:
        sr = seal['stages'][s]; sm = document(sr['stage_manifest']); oldstage = document(oldmain['stages'][s]['stage_manifest'])
        ex = document(sr['execution']); c.need(ex['exit_code'] == 0 and ex.get('failure') is None, 'Stage process incomplete')
        c.need(sm['stage'] == s and sm['status'] == 'native_average_stage_predictions_sealed' and sm['plan_sha256'] == PLAN_SHA, 'Stage contract differs')
        c.need(sm['request'] == seal['request'] and sm['protected_before'] == sm['protected_after'] == request['protected_sha256'], 'Stage provenance differs')
        c.need(sm['all_query_reload_exact'] is True and all(sm[k] is True for k in REPLAY_FLAGS), 'Stage endpoint replay/reload incomplete')
        c.need(all(sm[k] == 0 for k in ZERO_COUNTS) and all(sm[k] is False for k in
               ('query_labels_available', 'query_scores_computed', 'raw_price_pickle_opened')), 'Stage fit/label boundary differs')
        for name, digest in sm['artifact_sha256'].items(): bind(name, digest)
        start, end, nf, nq = c.SPLITS[s]
        c.need((sm['cutoff'], sm['query_end'], sm['outer_fit_rows'], sm['query_rows']) == (start, end, nf, nq), 'Fixed window differs')
        c.need(sm['source_R102_stage'] == oldmain['stages'][s]['stage_manifest'], 'Endpoint source differs')
        ids = {k: array(sm[k]) for k in ('fit_indices', 'query_indices', 'fit_source_cache_indices', 'query_source_cache_indices')}
        for k, a in ids.items(): c.exact(a, array(oldstage[k]), 'Original index differs: ' + k)
        fi, qi = ids['fit_indices'], ids['query_indices']
        metric.check_temporal(s, fi, qi, ids['query_source_cache_indices'], days, mapping)
        c.exact(mapping[fi], ids['fit_source_cache_indices'], 'Raw fit mapping differs')
        fixed = {k: array(rec) for k, rec in sm['fixed_branches'].items()}
        c.need(set(fixed) == {'area', 'lgb', 'H', 'U'}, 'Fixed branches differ')
        for k, a in fixed.items(): c.exact(a, array(oldstage['fixed_branches'][k]), 'Fixed branch changed')
        predictions = {k: array(rec) for k, rec in sm['predictions'].items()}
        layers = {arm: {k: array(rec) for k, rec in records.items()} for arm, records in sm['layers'].items()}
        c.need(set(predictions) == set(layers) == {'v22', 'C', 'T'} and sr['predictions'] == sm['predictions'], 'Arm identities differ')
        c.need(all(set(a) == set(LAYERS[:-1]) for a in layers.values()), 'Four premedian layers required')
        for a in [*fixed.values(), *predictions.values(), *[a for layer in layers.values() for a in layer.values()]]:
            metric.prices(a, (nq,), 'Saved query total')
        check_native_mean(layers); metric.check_formulas(predictions, layers, fixed)
        for arm, oldarm in (('v22', 'v22'), ('C', 'T')):
            c.exact(predictions[arm], array(oldstage['predictions'][oldarm]), 'Endpoint median differs')
            for k, a in layers[arm].items(): c.exact(a, array(oldstage['layers'][oldarm][k]), 'Endpoint layer differs: ' + arm + '/' + k)
        stages[s] = {'fit': Xraw.iloc[mapping[fi]].reset_index(drop=True), 'query': Xraw.iloc[mapping[qi]].reset_index(drop=True),
                     'query_indices': qi, 'query_source_cache_indices': ids['query_source_cache_indices'],
                     'predictions': predictions, 'layers': layers}
    c.need(request['protected_sha256'] == c.protected_hashes(), 'Protected changed during prelabel gate')
    return {'stages': stages, 'input_sha256': checked, 'source_sha256': own,
            'protected_before': request['protected_sha256'], 'raw_train': seal['raw_train'],
            'old_results': oldresults, 'sealed_manifest': c.record(sealed_path), 'evaluation_request': c.record(request_path)}


def run(sealed_path, request_path, outpath):
    out = c.safe(outpath); out.mkdir(parents=True, exist_ok=False); tick = time.monotonic(); failure = None; opened = False
    try:
        checked = preflight(sealed_path, request_path)
        c.write(out / 'prelabel_gate.json', {'status': 'three_candidates_and_both_endpoints_sealed_before_scoring',
                'input_sha256': checked['input_sha256'], 'source_sha256': checked['source_sha256'],
                'sealed_manifest': checked['sealed_manifest'], 'native_models_NPZ_or_R_S_states_deserialized': False,
                'new_fits': 0, 'completed_at_utc': c.now()})
        c.need(c.sha(checked['raw_train']['path']) == checked['raw_train']['sha256'], 'Raw prices changed')
        raw = pd.read_pickle(c.safe(checked['raw_train']['path'])); opened = True; stages = {}
        for s in c.STAGES:
            a = checked['stages'][s]; y = metric.select_query_y(raw, a['query_source_cache_indices'])
            score = lambda v: metric.descriptive_metrics(y, v, a['fit'], a['query'], c.SPLITS[s][0])
            metrics = {k: score(v) for k, v in a['predictions'].items()}
            layers = {arm: {k: score(v) for k, v in layer.items()} for arm, layer in a['layers'].items()}
            for arm in layers: layers[arm]['median'] = metrics[arm]
            old = checked['old_results']['stages'][s]
            for arm, oldarm in (('v22', 'v22'), ('C', 'T')):
                c.need(metrics[arm] == old['metrics'][oldarm] and layers[arm] == old['layer_metrics'][oldarm], 'Endpoint full five-layer metrics replay differs')
            c.need(c.values_sha(y) == old['query_target_values_sha256'], 'Original query target mapping differs')
            folder = out / s; folder.mkdir(); boots = {}; replicates = {}
            for control in ('v22', 'C'):
                item, samples = metric.paired(y, a['predictions'], a['fit'], a['query'], a['query_indices'], s, control)
                item['caution'] = CAUTION
                for family in item['methods'].values():
                    for short, key in (('overall', 'overall_mae'), ('weighted', 'weighted_mae')):
                        z = family['summary'][short]
                        c.need(abs(z['control_mae'] - metrics[control][key]) <= 1e-6 and
                               abs(z['candidate_mae'] - metrics['T'][key]) <= 1e-6, 'Bootstrap point differs')
                boots[control] = item
                replicates[control] = {f: c.save_array(folder / f'T_vs_{control}_{f}_replicates.npy', v) for f, v in samples.items()}
            layer_comparisons = {control: {k: metric.comparisons({arm: layers[arm][k] for arm in layers}, control)
                                           for k in LAYERS} for control in ('v22', 'C')}
            item = {'stage': s, 'cutoff': c.SPLITS[s][0], 'query_end': c.SPLITS[s][1],
                    'outer_fit_rows': len(a['fit']), 'query_rows': len(y), 'metrics': metrics, 'layer_metrics': layers,
                    'comparison': metric.comparisons(metrics), 'comparison_vs_C': metric.comparisons(metrics, 'C'),
                    'C_vs_v22': metric.comparisons(metrics, 'v22', 'C'), 'layer_comparisons': layer_comparisons,
                    'layer_comparisons_are_descriptive_not_adoption_gates': True,
                    'paired': boots, 'replicate_artifacts': replicates, 'control_C_role': 'saved complete R102T endpoint',
                    'query_target_values_sha256': c.values_sha(y), 'query_target_values_persisted': False,
                    'group_membership_uses_original_outer_fit': True, 'both_endpoint_five_layer_metrics_replay_exact': True}
            c.write(folder / 'results.json', item); stages[s] = item
            print(json.dumps({'stage': s, 'T_minus_v22': item['comparison']['overall_mae']['delta'],
                              'T_minus_C': item['comparison_vs_C']['overall_mae']['delta']}), flush=True)
        del raw
        c.verify_hashes(checked['input_sha256'])
        c.need(source_hashes() == checked['source_sha256'] and c.protected_hashes() == checked['protected_before'], 'Sources/protected changed')
        result = {'round': 105, 'status': 'completed_three_window_evaluation', 'created_at_utc': c.now(), 'parameters': parameters(),
                  'stages': stages, 'research_gates': research_gates(stages),
                  **{k: checked[k] for k in ('input_sha256', 'source_sha256', 'protected_before', 'sealed_manifest', 'evaluation_request')},
                  'protected_after': c.protected_hashes(), 'raw_pickle_whole_table_deserialized_after_all_seals': True,
                  'native_models_NPZ_or_R_S_states_deserialized': False, 'conditional_intervals_reported': 24,
                  'additional_bootstrap_calls': 0, 'new_fits': 0, 'model_load_predict_calls': 0, 'test_files_read': 0,
                  'new_CSV': False, 'selected_submission': 'v22', 'Private_score': None, 'fresh_blind_validation': False,
                  'variance_reduction_measured': False, 'fully_historically_as_of': False, 'caution': CAUTION,
                  'elapsed_seconds': time.monotonic() - tick}
        c.write(out / 'results.json', result); return result
    except Exception:
        failure = traceback.format_exc(); raise
    finally:
        c.write(out / 'execution.json', {'exit_code': int(failure is not None), 'failure': failure,
                'elapsed_seconds': time.monotonic() - tick, 'source_sha256': c.sha(__file__),
                'raw_prices_deserialized': opened, 'new_fits': 0, 'model_load_predict_calls': 0, 'automatic_retry': False})


def self_check():
    tick = time.monotonic(); checks = []
    def mark(name, ok): c.need(ok, name); checks.append(name)
    def reject(name, fn):
        try: fn()
        except (ValueError, KeyError): checks.append(name)
        else: raise AssertionError(name)
    stages = {s: {'metrics': {a: {'overall_mae': v, 'weighted_mae': v} for a, v in [('v22', 100.), ('C', 95.), ('T', 90.)]},
                  'comparison': {'quarter_or_large_group_deterioration_above_5pct': []}} for s in c.STAGES}
    mark('both_fixed_gates_pass', research_gates(stages)['both_research_gates_passed'])
    for name in ('overall_mae', 'weighted_mae'):
        z = copy.deepcopy(stages); z['auxiliary']['metrics']['T'][name] = 95.
        mark('equal_endpoint_not_improvement_' + name, not research_gates(z)['supplementary_endpoint']['supplementary_endpoint_gate_passed'])
    z = copy.deepcopy(stages); z['confirmation']['metrics']['T']['overall_mae'] = 99.6
    mark('latest_half_percent_required', not research_gates(z)['primary']['adoption_research_gate_passed'])
    z = copy.deepcopy(stages); z['development']['comparison']['quarter_or_large_group_deterioration_above_5pct'] = [{}]
    mark('fixed_group_adverse_required', not research_gates(z)['primary']['adoption_research_gate_passed'])
    mark('native_is_not_gate', research_gates(stages)['supplementary_endpoint']['native_improvement_required'] is False)
    z = {'v22': {'native': np.array([1., 4., 20.])}, 'C': {'native': np.array([9., 2., 30.])},
         'T': {'native': np.array([5., 3., 25.])}}
    check_native_mean(z); checks.append('float64_native_mean_formula')
    bad = copy.deepcopy(z); bad['T']['native'] = .25 * z['v22']['native'] + .75 * z['C']['native']
    reject('different_alpha_rejected', lambda: check_native_mean(bad))
    header = {'schema_version': 1, 'status': 'all_native_average_predictions_sealed', 'stage_order': list(c.STAGES),
              'stages': dict.fromkeys(c.STAGES), 'plan_sha256': PLAN_SHA, 'all_query_reload_exact': True,
              'protected_before': dict.fromkeys(range(9), 'x'), 'protected_after': dict.fromkeys(range(9), 'x'),
              **dict.fromkeys(ZERO_COUNTS, 0), **dict.fromkeys(REPLAY_FLAGS, True),
              **dict.fromkeys(('query_labels_available', 'query_scores_computed', 'raw_price_pickle_opened'), False)}
    check_header(header); checks.append('zero_fit_complete_header')
    for k, v in [('new_native_fits', 1), ('query_scores_computed', True), ('all_endpoints_replay_exact', False), ('plan_sha256', c.PLAN_SHA)]:
        reject('reject_header_' + k, lambda k=k, v=v: check_header({**header, k: v}))
    mark('fixed_replicates_and_roles', parameters()['replicates'] == 2000 and parameters()['seed'] == 20260925 and
         parameters()['stage_indices'] == {'development': 0, 'confirmation': 1, 'auxiliary': 2} and
         parameters()['family_indices'] == {'project': 0, 'month': 1} and parameters()['endpoint_control'] == 'C')
    return {'status': 'passed', 'exit_code': 0, 'check_count': len(checks), 'checks': checks,
            'source_sha256': c.sha(__file__), 'source_hashes': source_hashes(), 'plan_sha256': PLAN_SHA,
            'parameters': parameters(), 'real_data_read': False, 'new_fit_score_calls': 0,
            'elapsed_seconds': time.monotonic() - tick}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--self-check', action='store_true'); ap.add_argument('--receipt')
    ap.add_argument('--sealed-manifest'); ap.add_argument('--evaluation-request'); ap.add_argument('--out')
    a = ap.parse_args()
    if a.self_check:
        result = self_check()
        if a.receipt: c.write(a.receipt, result)
        print(json.dumps(result, ensure_ascii=False), flush=True)
    else:
        if not all((a.sealed_manifest, a.evaluation_request, a.out)): ap.error('sealed-manifest/evaluation-request/out required')
        run(a.sealed_manifest, a.evaluation_request, a.out)
