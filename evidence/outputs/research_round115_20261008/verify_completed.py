"""Independent bounded R115 aggregate/identity audit; never deserialize X/arrays/prices."""
import ast
import importlib.util
import json
import math
from pathlib import Path
import sys
import time
from unittest.mock import patch
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import round100_contract as c
import round115_prior_support_x_audit as q

FOLDER = 'outputs/research_round115_20261008/'
OUT = 'outputs/round115/'
checks = []


def need(name, condition):
    c.need(bool(condition), name)
    checks.append(name)


def close(a, b):
    return math.isclose(float(a), float(b), rel_tol=3e-12, abs_tol=3e-8)


def locked(name):
    value = c.read(name)
    sidecar = c.safe(str(Path(name).with_suffix('.sha256')))
    need('Sidecar exact ' + name, sidecar.read_text(encoding='ascii').strip() == c.sha(name))
    return value


def distribution(value, rows=None, bounds=None):
    need('Distribution keys', set(value) == {'rows', 'mean', 'min', 'max', 'quantiles'})
    n = value['rows']
    need('Nonnegative distribution support', type(n) is int and n >= 0)
    if rows is not None:
        need('Distribution population count', n == rows)
    quant = value['quantiles']
    need('Fixed linear quantiles', set(quant) == set(map(str, q.QUANTILES)))
    data = [quant[str(x)] for x in q.QUANTILES]
    if n == 0:
        need('Empty distributions explicit', value['mean'] is None and value['min'] is None
             and value['max'] is None and all(x is None for x in data))
        return
    need('Finite ordered aggregate statistics', all(type(v) in (int, float) and math.isfinite(v)
         for v in [value['mean'], value['min'], value['max'], *data]))
    need('Quantile and endpoint monotonicity', all(a <= b for a, b in zip(data, data[1:]))
         and data[0] == value['min'] and data[-1] == value['max']
         and value['min'] - 1e-9 <= value['mean'] <= value['max'] + 1e-9)
    if bounds is not None:
        need('Distribution domain', value['min'] >= bounds[0] - 1e-12 and value['max'] <= bounds[1] + 1e-12)


def metric_distributions(metrics, rows=None):
    for key, value in metrics.items():
        bounds = (0., 1.) if 'share' in key else (0., math.inf)
        distribution(value, rows, bounds)
    if not metrics or next(iter(metrics.values()))['rows'] == 0:
        return
    for field in ('mean', 'min', 'max'):
        need('ESS no greater than project count aggregate', metrics['date_weight_project_ESS'][field]
             <= metrics['project_groups'][field] + 1e-7)
        need('Project count no greater than raw support aggregate', metrics['project_groups'][field]
             <= metrics['raw_rows'][field] + 1e-7)
    for quantile in map(str, q.QUANTILES):
        need('ESS/project/raw ordered quantile domains',
             metrics['date_weight_project_ESS']['quantiles'][quantile]
             <= metrics['project_groups']['quantiles'][quantile] + 1e-7
             <= metrics['raw_rows']['quantiles'][quantile] + 2e-7)


def summary(value, expected=None, weighted=False):
    n = value['rows']
    if expected is not None:
        need('Routed population identity', n == expected)
    levels = value['selected_level_rows']
    need('All five level partitions', set(levels) == set(map(str, range(5)))
         and all(type(v) is int and v >= 0 for v in levels.values()) and sum(levels.values()) == n)
    need('All four candidate supports', set(value['four_level_raw_counts']) == set(map(str, range(4))))
    for d in value['four_level_raw_counts'].values():
        distribution(d, n, (0., math.inf))
    metrics = value['selected_support_and_exposure']
    need('Fixed selected support/exposure metrics', set(metrics) == set(q.METRICS) | {
        'same_cluster_peers_reference', 'same_cluster_peers_selected_cell', 'same_cluster_selected_row_share'})
    metric_distributions(metrics, n)
    if n:
        need('Selected reference nonempty', metrics['raw_rows']['min'] >= 1
             and metrics['project_groups']['min'] >= 1 and metrics['date_weight_project_ESS']['min'] >= 1 - 1e-9)
        need('Selected-cell peer mean bounded by all-reference peer mean',
             metrics['same_cluster_peers_selected_cell']['mean'] <= metrics['same_cluster_peers_reference']['mean'] + 1e-9)
    if weighted:
        masses = value['selected_level_date_weight_mass']
        need('All five date-mass partitions', set(masses) == set(map(str, range(5)))
             and all(math.isfinite(v) and v >= 0 for v in masses.values())
             and close(sum(masses.values()), value['saved_date_weight_mass']))
    # Statistical aggregates do not carry raw keys, identifiers or row vectors.
    def numeric_only(v):
        if isinstance(v, dict):
            return all(numeric_only(x) for x in v.values())
        return type(v) in (int, float, bool) or v is None
    need('Numeric-only statistical payload', numeric_only(value))


def partition(total, parts, weighted=False):
    need('Partition count total', sum(v['rows'] for v in parts) == total['rows'])
    for level in map(str, range(5)):
        need('Partition selected-level count', sum(v['selected_level_rows'][level] for v in parts)
             == total['selected_level_rows'][level])
        if weighted:
            need('Partition selected-level date mass', close(sum(v['selected_level_date_weight_mass'][level] for v in parts),
                 total['selected_level_date_weight_mass'][level]))
    if weighted:
        need('Partition total saved date mass', close(sum(v['saved_date_weight_mass'] for v in parts), total['saved_date_weight_mass']))
    for family in ('four_level_raw_counts', 'selected_support_and_exposure'):
        for name, target in total[family].items():
            mass = sum(v[family][name]['mean'] * v['rows'] for v in parts if v['rows'])
            need('Partition aggregate means ' + family + '/' + name, close(mass / total['rows'], target['mean']))


def cells(items):
    need('Four original cell levels', len(items) == 4 and [v['level'] for v in items] == list(range(4)))
    for level in items:
        for population in ('all_cells', 'supported_cells'):
            metrics = level[population]
            need('Fixed cell concentration metrics', set(metrics) == set(q.METRICS))
            counts = {v['rows'] for v in metrics.values()}
            need('Every cell metric uses same cell population', len(counts) == 1)
            metric_distributions(metrics)
            if population == 'supported_cells' and next(iter(counts)):
                need('Supported cells have at least30 raw transactions', metrics['raw_rows']['min'] >= 30)
        need('Supported cell subset size', level['supported_cells']['raw_rows']['rows'] <= level['all_cells']['raw_rows']['rows'])


def global_stat(value, expected):
    need('Fixed global support identity', set(value) == set(q.METRICS) and value['raw_rows'] == expected
         and all(math.isfinite(v) for v in value.values())
         and 1 <= value['date_weight_project_ESS'] <= value['project_groups'] + 1e-8 <= expected + 1e-8
         and 0 < value['largest_project_row_share'] <= 1
         and 0 < value['largest_project_date_mass_share'] <= 1)


def execution(name, mode, source=True):
    value = c.read(FOLDER + name + '_execution.json')
    expected = [str(ROOT / '.venv/Scripts/python.exe'), '-u', '-B', '-X', 'utf8',
                str(ROOT / 'src/round115_prior_support_x_audit.py'), '--' + mode]
    need('Formal ' + mode + ' exit/command', value['exit_code'] == 0 and value['command'] == expected
         and value['new_fits'] == value['GPU_calls'] == 0)
    return value


def main():
    tick = time.monotonic()
    # Fail if this independent audit or re-executed tiny arithmetic attempts any data loading.
    def forbidden(*args, **kwargs):
        raise AssertionError('Independent audit may not deserialize housing frames, arrays or labels')
    with patch.object(q.pd, 'read_pickle', forbidden), patch.object(np, 'load', forbidden):
        spec = q.specification()
        source = {**spec['source_sha256'], q.PLAN: spec['plan']['sha256']}
        q.verify_hashes(source); q.verify_hashes(spec['input_sha256'])
        need('Exact30 frozen metadata/target-free readset', len(spec['input_sha256']) == 30 and not spec['blockers'])
        preflight = locked(FOLDER + 'preflight_review.json')
        q.validate_preflight(preflight, spec)
        request = locked(OUT + 'request.json'); started = locked(OUT + 'started.json'); results = locked(OUT + 'results.json')
        need('Full immutable request identity', request['specification'] == spec and request['source_sha256'] == spec['source_sha256']
             and request['plan'] == spec['plan'] and request['input_sha256'] == spec['input_sha256']
             and request['parameters'] == q.parameters() and request['independent_preflight'] == c.record(q.PREFLIGHT))
        need('Request before data deserialization', request['status'] == 'pure_X_prior_support_audit_frozen_before_data_deserialization'
             and request['real_X_or_arrays_deserialized'] is False and request['input_byte_hashes_verified_at_freeze'] is True
             and request['new_fits'] == request['GPU_calls'] == 0 and request['query_labels_read'] is False)
        need('Started exact request before data', started['request'] == c.record(OUT + 'request.json')
             and started['real_X_or_arrays_deserialized'] is False and started['new_fits'] == 0)
        execution('freeze', 'freeze'); run_execution = execution('run', 'run')
        freeze_log = json.loads(c.safe(FOLDER + 'freeze.log').read_text(encoding='utf8'))
        need('Formal freeze exact request/count', freeze_log['request'] == c.record(OUT + 'request.json')
             and freeze_log['input_files'] == 30 and freeze_log['new_fits'] == 0)
        run_lines = [json.loads(line) for line in c.safe(FOLDER + 'run.log').read_text(encoding='utf8').splitlines() if line]
        need('Formal run logs all four fixed stages and completion', [v['stage'] for v in run_lines[:-1]] == list(q.STAGES)
             and run_lines[-1]['status'] == results['status'] and run_lines[-1]['elapsed_seconds'] == results['elapsed_seconds'])
        need('Formal run wrapper/internal timing', 0 < results['elapsed_seconds'] <= run_execution['elapsed_seconds'])
        for stem, count, expected_command, fn in (
            ('formal_author_selfcheck', 41, [str(ROOT / '.venv/Scripts/python.exe'), '-B', '-X', 'utf8', str(ROOT / 'src/round115_prior_support_x_audit.py'), '--self-check'], q.self_check),
            ('root_independent_checks', 75, [str(ROOT / '.venv/Scripts/python.exe'), '-B', '-X', 'utf8', str(ROOT / FOLDER / 'independent_preflight_checks.py')], None)):
            receipt = locked(FOLDER + stem + '_execution.json')
            log = json.loads(c.safe(FOLDER + stem + '.log').read_text(encoding='utf8'))
            need('Formal synthetic receipt and log ' + stem, receipt['exit_code'] == 0 and receipt['command'] == expected_command
                 and receipt['check_count'] == log['check_count'] == count and log['status'] == 'passed'
                 and receipt['source_sha256'] == spec['source_sha256'] and receipt['log_sha256'] == c.sha(FOLDER + stem + '.log'))
            if fn is None:
                module_spec = importlib.util.spec_from_file_location('independent_r115_root_arithmetic', c.safe(FOLDER + 'independent_preflight_checks.py'))
                module = importlib.util.module_from_spec(module_spec); module_spec.loader.exec_module(module)
                fn = module.run
            actual = fn()
            need('Independent re-execution of synthetic checks only ' + stem,
                 actual['status'] == 'passed' and actual['check_count'] == count)
        need('Complete result source/plan/request/preflight bindings', results['source_sha256'] == spec['source_sha256']
             and results['input_sha256'] == spec['input_sha256'] and results['plan'] == spec['plan']
             and results['parameters'] == q.parameters() and results['request'] == c.record(OUT + 'request.json')
             and results['independent_preflight'] == c.record(q.PREFLIGHT)
             and set(results['stages']) == set(q.STAGES))
        expected_opened = {q.FEATURES, q.MAPPING}
        for origin in spec['origins'].values(): expected_opened.update(origin['paths'].values())
        need('Exact21 typed deserialized artifacts, no extra data load', results['deserialized_artifacts'] == sorted(expected_opened)
             and len(expected_opened) == 21 and all(Path(v).suffix in ('.pkl', '.npy') for v in expected_opened)
             and results['full_target_free_cache_deserialized'] is True)
        for name in ('new_fits', 'GPU_calls', 'MAE_calls', 'bootstrap_calls', 'model_loads', 'price_prior_builds'):
            need('Zero diagnostic scope ' + name, results[name] == 0)
        for name in ('query_labels_read', 'raw_price_pickle_opened', 'NPZ_opened', 'new_CSV', 'automatic_model_experiment_authorized'):
            need('No forbidden diagnostic action ' + name, results[name] is False)
        need('No Public/Private result', results['Public_score'] is None and results['Private_score'] is None)
        need('Actual input/source after-run invariant', results['inputs_and_sources_unchanged'] is True)
        stage_summary = {}; artifacts = {}
        for stage in q.STAGES:
            name = OUT + stage + '/aggregates.json'; a = locked(name)
            need('Exact immutable aggregate identity ' + stage, results['stages'][stage] == c.record(name)
                 and a['request'] == c.record(OUT + 'request.json') and a['source_sha256'] == spec['source_sha256']
                 and a['plan'] == spec['plan'] and a['parameters'] == q.parameters() and a['selected_origin_source'] == spec['origins'][stage]
                 and a['stage'] == stage and a['round'] == 115 and a['status'] == 'pure_X_support_exposure_completed'
                 and a['cutoff'] == q.WINDOWS[stage][0] and a['query_end'] == q.WINDOWS[stage][1])
            nf, nq = q.WINDOWS[stage][2:]
            need('Exact fit/query count', (a['fit_rows'], a['query_rows']) == (nf, nq))
            for role in ('fit_indices', 'query_indices'):
                d = a['input_identities'][role]
                need('Canonical index aggregate identity', d == {'dtype': 'int64', 'shape': [nf if role == 'fit_indices' else nq],
                     'values_sha256': spec['origins'][stage]['index_contract'][role + '_content_sha256']})
            need('Saved dateweight aggregate identity', a['input_identities']['date_weights'] == {'dtype': 'float32', 'shape': [nf],
                 'values_sha256': spec['origins'][stage]['weights_values_sha256']})
            for kind in ('group_fit', 'transaction_fit'): summary(a[kind], nf, True)
            for label in ('fit_named_project', 'fit_missing_project_proxy'):
                for policy in ('group', 'transaction'): summary(a[label][policy], weighted=True)
            for policy, all_key in (('group', 'group_fit'), ('transaction', 'transaction_fit')):
                partition(a[all_key], [a['fit_named_project'][policy], a['fit_missing_project_proxy'][policy]], True)
            peers = a['group_fit']['selected_support_and_exposure']
            for key in ('same_cluster_peers_reference', 'same_cluster_peers_selected_cell', 'same_cluster_selected_row_share'):
                need('Zero group-policy same-cluster peers', peers[key]['max'] == peers[key]['min'] == peers[key]['mean'] == 0
                     and all(v == 0 for v in peers[key]['quantiles'].values()))
            summary(a['full_fit_query'], nq)
            for family in ('query_by_project_status', 'query_by_city', 'query_by_forecast_quarter'):
                for value in a[family].values(): summary(value)
                partition(a['full_fit_query'], list(a[family].values()))
            need('All three statuses/all four quarters', set(a['query_by_project_status']) == {'seen', 'new_nonmissing', 'missing'}
                 and set(a['query_by_forecast_quarter']) == set(map(str, range(1, 5))))
            transitions = a['selected_level_transitions']
            need('Exact25 transitions including empty cells', set(transitions) == {f'{i}->{j}' for i in range(5) for j in range(5)}
                 and sum(v['rows'] for v in transitions.values()) == nf)
            for i in range(5):
                need('Transition row/column count margins', sum(transitions[f'{i}->{j}']['rows'] for j in range(5)) == a['group_fit']['selected_level_rows'][str(i)]
                     and sum(transitions[f'{j}->{i}']['rows'] for j in range(5)) == a['transaction_fit']['selected_level_rows'][str(i)])
                need('Transition row/column date-mass margins', close(sum(transitions[f'{i}->{j}']['saved_date_weight_mass'] for j in range(5)), a['group_fit']['selected_level_date_weight_mass'][str(i)])
                     and close(sum(transitions[f'{j}->{i}']['saved_date_weight_mass'] for j in range(5)), a['transaction_fit']['selected_level_date_weight_mass'][str(i)]))
            need('Transition total date mass', close(sum(v['saved_date_weight_mass'] for v in transitions.values()), a['group_fit']['saved_date_weight_mass']))
            distribution(a['raw_support_transaction_over_group'], nf, (0., math.inf))
            # Only the seed42 index permutation is recreated: no housing array is opened.
            tf = q.transaction_folds(nf)
            need('Fixed transaction permutation identity and balanced sizes', a['transaction_fold_identity'] == q.identity(tf)
                 and a['transaction_fold_rows'] == [int((tf == j).sum()) for j in range(5)]
                 and max(a['transaction_fold_rows']) - min(a['transaction_fold_rows']) <= 1)
            for policy, counts_key in (('group', 'original_group_fold_rows'), ('transaction', 'transaction_fold_rows')):
                sizes = a[counts_key]; folds = a['fold_reference_summaries'][policy]
                need('Fold full partition', len(sizes) == len(folds) == 5 and sum(sizes) == nf)
                for i, fold in enumerate(folds):
                    need('Fold held/reference complement', fold['fold'] == i and fold['held_rows'] == sizes[i] and fold['reference_rows'] == nf - sizes[i])
                    if sizes[i]:
                        global_stat(fold['reference_global'], nf - sizes[i]); cells(fold['reference_cell_summary'])
            cells(a['full_fit_query_cell_summary']); global_stat(a['full_fit_query_global'], nf)
            for flag in ('same_query_reference_for_both_policies', 'inputs_unchanged', 'raw_rows_are_not_independent_projects', 'project_mass_ESS_is_weight_concentration_only'):
                need('Scope preserved ' + flag, a[flag] is True)
            for flag in ('prices_used', 'query_labels_read', 'automatic_model_experiment_authorized'):
                need('No stage price/model action ' + flag, a[flag] is False)
            need('No stage Public/Private', a['Public_score'] is None and a['Private_score'] is None)
            offdiag = sum(v['rows'] for k, v in transitions.items() if k.split('->')[0] != k.split('->')[1])
            finer = sum(v['rows'] for k, v in transitions.items() if int(k.split('->')[1]) < int(k.split('->')[0]))
            coarser = offdiag - finer
            need('Structural flags exactly reflect aggregates', a['structural_presence_not_model_gate']['any_selected_level_changes'] == (offdiag > 0)
                 and a['structural_presence_not_model_gate']['any_transaction_same_cluster_exposure'] == (a['transaction_fit']['selected_support_and_exposure']['same_cluster_peers_reference']['max'] > 0))
            stage_summary[stage] = {'fit_rows': nf, 'query_rows': nq, 'changed_level_rows': offdiag,
                'changed_level_fraction': offdiag / nf, 'to_finer_rows': finer, 'to_coarser_rows': coarser,
                'median_transaction_over_group_support': a['raw_support_transaction_over_group']['quantiles']['0.5'],
                'selected_project_mass_ESS_median': {k: a[k]['selected_support_and_exposure']['date_weight_project_ESS']['quantiles']['0.5'] for k in ('group_fit', 'transaction_fit')},
                'transaction_same_cluster_peer_mean': a['transaction_fit']['selected_support_and_exposure']['same_cluster_peers_reference']['mean'],
                'query_status_rows': {k: v['rows'] for k, v in a['query_by_project_status'].items()},
                'elapsed_seconds': a['elapsed_seconds']}
            artifacts[stage] = c.record(name)
        q.verify_hashes(source); q.verify_hashes(spec['input_sha256'])
        need('No failure/overwrite evidence', not c.safe(OUT + 'failure.json').exists())
    payload = {'status': 'passed', 'created_at_utc': c.now(), 'round': 115,
        'scope': 'Independent source, frozen readset, synthetic arithmetic, actual aggregate consistency/privacy audit; not row-by-row independent housing statistics replay.',
        'source_sha256': source, 'request': c.record(OUT + 'request.json'), 'started': c.record(OUT + 'started.json'),
        'results': c.record(OUT + 'results.json'), 'preflight_review': c.record(q.PREFLIGHT), 'plan': spec['plan'],
        'freeze_execution': c.record(FOLDER + 'freeze_execution.json'), 'run_execution': c.record(FOLDER + 'run_execution.json'),
        'formal_author_selfcheck': c.record(FOLDER + 'formal_author_selfcheck_execution.json'),
        'root_independent_arithmetic': c.record(FOLDER + 'root_independent_checks_execution.json'),
        'review_tool': c.record(__file__), 'artifacts': artifacts, 'stage_summary': stage_summary,
        'checks_executed': len(checks), 'all_checks': sorted(set(checks)), 'input_files_stream_hashed_without_deserialization': 30,
        'declared_actual_target_free_deserialized_artifacts': 21, 'independent_synthetic_reexecutions': {'author': 41, 'root_manual_arithmetic': 75},
        'source_and_input_before_after_verified': True, 'aggregate_privacy_numeric_only_statistics_verified': True,
        'independent_full_X_statistics_recomputed': False, 'housing_X_or_arrays_deserialized_by_reviewer': False,
        'prices_or_query_y_read_by_reviewer': False, 'new_model_fits': 0, 'GPU_work': False, 'new_CSV': False,
        'Public_score': None, 'Private_score': None, 'blocking_findings_remaining': [],
        'research_recommendation': 'A separate single fixed transaction-fold-prior comparison is defensible as a hypothesis test; no training or adoption authorized by this receipt.',
        'limits': ['All actual row-level statistics rely on frozen run; reviewer checked complete aggregate partitions/margins/means/domains and synthetic independently calculated references, not every housing row.',
                   'Actual opened-artifact inventory is frozen-source execution evidence, not OS-level syscall tracing. Full pure-X cache was deserialized in the approved run.',
                   'Project/fallback groups are proxies; ESS measures date-mass concentration, not independent project/sample count or price reliability.',
                   'Route changes and same-project exposure do not establish prior price change, MAE improvement, Private benefit, or permission to fit.',
                   'Four origins overlap and are reused; no blind four-vote interpretation.'],
        'review_elapsed_seconds': time.monotonic() - tick}
    c.write(FOLDER + 'postfit_review.json', payload)
    print(json.dumps({'status': 'passed', 'review': c.record(FOLDER + 'postfit_review.json'), 'checks': len(checks)}, ensure_ascii=False))


if __name__ == '__main__':
    main()
