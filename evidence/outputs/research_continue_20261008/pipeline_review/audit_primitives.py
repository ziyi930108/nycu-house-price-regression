"""Independent arithmetic primitives; no data access, ML imports, or fit on import."""
from __future__ import annotations

import math


def need(ok, message):
    if not ok:
        raise ValueError(message)


def file_identity(record):
    need(isinstance(record, dict) and isinstance(record.get('path'), str)
         and isinstance(record.get('sha256'), str) and len(record['sha256']) == 64,
         'Malformed file identity')
    return {key: record[key] for key in ('path', 'sha256')}


def manual_comparison(control, candidate, control_name):
    """Rebuild point/group arithmetic, without calling a research driver gate."""
    def delta(a, b):
        need(math.isfinite(a) and math.isfinite(b) and a >= 0 and b >= 0,
             'Nonfinite or negative MAE')
        return {'control_mae': a, 'candidate_mae': b, 'delta': b-a,
                'relative_change_pct': 100*(b-a)/a if a else None}

    out = {k: delta(control[k], candidate[k]) for k in ('overall_mae', 'weighted_mae')}
    adverse = []
    for section in ('strata', 'forecast_quarters', 'months', 'cities', 'relative_forecast_months'):
        need(set(control[section]) == set(candidate[section]), 'Group names differ')
        out[section] = {}
        for name, a in control[section].items():
            b = candidate[section][name]
            need(a['rows'] == b['rows'] and a['rows'] >= 0, 'Group support differs')
            if a['rows']:
                item = {'rows': a['rows'], **delta(a['mae'], b['mae'])}
                out[section][name] = item
                if (section == 'forecast_quarters' or a['rows'] >= 500) and b['mae'] > 1.05*a['mae']:
                    adverse.append({'section': section, 'group': name, **item})
    need(len(control['price_bands']) == len(candidate['price_bands']), 'Price band count differs')
    out['price_bands'] = []
    for a, b in zip(control['price_bands'], candidate['price_bands']):
        need((a['band'], a['rows']) == (b['band'], b['rows']), 'Price band support differs')
        item = {'band': a['band'], 'rows': a['rows'], **(delta(a['mae'], b['mae']) if a['rows'] else {})}
        out['price_bands'].append(item)
        if a['rows'] >= 500 and b['mae'] > 1.05*a['mae']:
            adverse.append({'section': 'price_bands', **item})
    out['both_point_metrics_improve'] = all(candidate[k] < control[k] for k in ('overall_mae', 'weighted_mae'))
    out['quarter_or_large_group_deterioration_above_5pct'] = adverse
    if control_name == 'v22':
        out['stage_research_gate_passed'] = out['both_point_metrics_improve'] and not adverse
    out.update(control=control_name, candidate='T', used_for_research_gate=control_name == 'v22')
    return out


def manual_gates(stages):
    need(set(stages) == {'development', 'confirmation', 'auxiliary'}, 'Exactly three stages required')
    latest = stages['confirmation']['metrics']
    metrics = [v['metrics'] for v in stages.values()]
    no_adverse = all(not v['comparison']['quarter_or_large_group_deterioration_above_5pct']
                     for v in stages.values())
    fraction = (latest['v22']['overall_mae']-latest['T']['overall_mae'])/latest['v22']['overall_mae']
    strong = {
        'all_three_overall_improve': all(m['T']['overall_mae'] < m['v22']['overall_mae'] for m in metrics),
        'all_three_weighted_improve': all(m['T']['weighted_mae'] < m['v22']['weighted_mae'] for m in metrics),
        'confirmation_at_least_half_percent': fraction >= .005,
        'no_quarter_or_large_group_deterioration_above_5pct': no_adverse,
    }
    trial = {
        'latest_both_no_worse': all(latest['T'][k] <= latest['v22'][k] for k in ('overall_mae', 'weighted_mae')),
        'each_metric_each_window_within_quarter_percent': all(
            m['T'][k] <= 1.0025*m['v22'][k] for m in metrics for k in ('overall_mae', 'weighted_mae')),
        'at_least_two_windows_overall_improve': sum(m['T']['overall_mae'] < m['v22']['overall_mae'] for m in metrics) >= 2,
        'no_large_group_or_quarter_above_five_percent': no_adverse,
    }
    return {'components': trial, 'strong_components': strong,
            'exploratory_submission_eligible': all(trial.values()),
            'strong_research_gate_passed': all(strong.values()),
            'confirmation_fractional_improvement': fraction}


def self_check():
    import copy
    def metric(overall=100., weighted=100.):
        item = {'overall_mae': overall, 'weighted_mae': weighted, 'price_bands': [
            {'band': 'x', 'rows': 500, 'mae': 100.}]}
        for section in ('strata', 'forecast_quarters', 'months', 'cities', 'relative_forecast_months'):
            item[section] = {'x': {'rows': 500, 'mae': 100.}}
        return item
    def stages(values):
        return {s: {'metrics': {'v22': metric(), 'T': metric(v, v)},
                    'comparison': manual_comparison(metric(), metric(v, v), 'v22')}
                for s, v in zip(('development', 'confirmation', 'auxiliary'), values)}
    checks = []
    def check(name, condition):
        need(condition, name)
        checks.append(name)
    a = {'path': 'x.npy', 'sha256': '0'*64, 'shape': [2], 'dtype': 'float64', 'values_sha256': '1'*64}
    check('array_record_file_projection', file_identity(a) == {'path': 'x.npy', 'sha256': '0'*64})
    check('exploratory_two_improvements_latest_equal', manual_gates(stages([99., 100., 99.]))['exploratory_submission_eligible'])
    check('only_one_improvement_rejected', not manual_gates(stages([100., 100., 99.]))['exploratory_submission_eligible'])
    check('latest_any_regression_rejected', not manual_gates(stages([99., 100.00001, 99.]))['exploratory_submission_eligible'])
    check('quarter_percent_endpoint_inclusive', manual_gates(stages([100.25, 99., 99.]))['exploratory_submission_eligible'])
    check('quarter_percent_excess_rejected', not manual_gates(stages([100.25001, 99., 99.]))['exploratory_submission_eligible'])
    check('half_percent_strong_endpoint', manual_gates(stages([99., 99.5, 99.]))['strong_research_gate_passed'])
    check('half_percent_excess_strong_rejected', not manual_gates(stages([99., 99.50001, 99.]))['strong_research_gate_passed'])
    a, b = metric(), metric()
    b['cities']['x']['mae'] = 105.
    check('five_percent_endpoint_allowed', not manual_comparison(a, b, 'v22')['quarter_or_large_group_deterioration_above_5pct'])
    b['cities']['x']['mae'] = 105.00001
    check('large_city_excess_flagged', len(manual_comparison(a, b, 'v22')['quarter_or_large_group_deterioration_above_5pct']) == 1)
    a['cities']['x']['rows'] = b['cities']['x']['rows'] = 499
    check('small_city_excluded_from_gate', not manual_comparison(a, b, 'v22')['quarter_or_large_group_deterioration_above_5pct'])
    a['forecast_quarters']['x']['rows'] = b['forecast_quarters']['x']['rows'] = 499
    b['forecast_quarters']['x']['mae'] = 105.00001
    check('quarter_all_support_checked', len(manual_comparison(a, b, 'v22')['quarter_or_large_group_deterioration_above_5pct']) == 1)
    check('secondary_control_label', manual_comparison(metric(), metric(99., 99.), 'v24')['used_for_research_gate'] is False)
    changed = copy.deepcopy(metric())
    changed['cities']['x']['rows'] = 501
    try:
        manual_comparison(metric(), changed, 'v22')
    except ValueError:
        checks.append('different_group_support_rejected')
    else:
        raise ValueError('Changed support silently accepted')
    return {'status': 'passed', 'checks': checks, 'check_count': len(checks),
            'new_fits': 0, 'GPU_calls': 0, 'query_labels_loaded': False,
            'price_data_loaded': False, 'synthetic_values_only': True}


if __name__ == '__main__':
    import json
    print(json.dumps(self_check(), ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False))
