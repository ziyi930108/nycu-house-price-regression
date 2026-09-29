"""Fixed R103 descriptive algebra; no data loader, model, fit or selection."""
import argparse
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path
import time
import traceback

import numpy as np

ARMS = ('v22', 'C', 'T')
LAYERS = ('native', 'base', 'reference', 'spatial', 'median')
ROUTES = ('below', 'lower_equal', 'inside', 'upper_equal', 'above', 'both_equal')
SIGNS = ('negative', 'zero', 'positive')
CHANGES = ('improved', 'equal', 'worsened')
TOL = 1e-5
ROOT = Path(__file__).resolve().parents[1]
PLAN_SHA = '9d59b91dbb3cc6dd1adef038a4ea68c307da024fb2fe908b20fde20f03a0955c'


def need(ok, message):
    if not ok:
        raise ValueError(message)


def parameters():
    return {'round': 103, 'arms': list(ARMS), 'layers': list(LAYERS),
            'comparisons': ['T-v22', 'T-C'], 'routes': list(ROUTES),
            'error_signs': list(SIGNS), 'changes': list(CHANGES),
            'base_formula': '.75*(.5*A+.5*P)+.25*L',
            'row_identity_tolerance_TWD': TOL, 'zero_sign_rule': 'exact_float64_zero',
            'weights_renormalized': False, 'new_fit_or_bootstrap': False}


def source_hashes():
    return {'src/round103_complementarity_math.py': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}


def vector(value, n, name):
    a = np.asarray(value)
    need(a.dtype == np.float64 and a.ndim == 1 and (n is None or len(a) == n)
         and np.isfinite(a).all(), name + ' must be a finite aligned float64 vector')
    return a


def route(n, lo, hi):
    masks = (n < lo, (lo < hi) & (n == lo), (lo < n) & (n < hi),
             (lo < hi) & (n == hi), n > hi, (lo == hi) & (n == lo))
    need(np.all(np.sum(masks, axis=0) == 1), 'Route partition failed')
    return dict(zip(ROUTES, masks))


def sign_masks(a, labels=SIGNS):
    return dict(zip(labels, (a < 0, a == 0, a > 0)))


def summarize(mask, quantities, weights):
    n = len(weights); rows = int(mask.sum()); mass = float(weights[mask].sum())
    out = {'rows': rows, 'row_fraction': rows/n, 'weight_mass': mass, 'quantities': {}}
    for name, a in quantities.items():
        total = float(a[mask].sum()); weighted = float(np.dot(weights[mask], a[mask]))
        out['quantities'][name] = {'ordinary_mean': total/rows if rows else None,
            'ordinary_contribution': total/n, 'weighted_mean': weighted/mass if mass > 0 else None,
            'weighted_contribution': weighted}
    return out


def check_sum(cells, whole, message):
    need(sum(c['rows'] for c in cells) == whole['rows'], message + ' row sum')
    need(abs(sum(c['weight_mass'] for c in cells)-whole['weight_mass']) < 1e-12, message + ' mass sum')
    for key, q in whole['quantities'].items():
        for metric in ('ordinary_contribution', 'weighted_contribution'):
            need(abs(sum(c['quantities'][key][metric] for c in cells)-q[metric]) <= TOL,
                 message + ' contribution sum: ' + key + '/' + metric)


def check_identity(record):
    q = record['quantities']
    for metric in ('ordinary_mean', 'ordinary_contribution', 'weighted_mean', 'weighted_contribution'):
        if q['base_delta'][metric] is not None:
            need(abs(q['base_delta'][metric]-q['native_term'][metric]-q['cancellation_term'][metric]) <= TOL,
                 'Aggregated base identity failed')
            need(abs(q['median_delta'][metric]-q['spatial_delta'][metric]-q['clipping_term'][metric]) <= TOL,
                 'Aggregated median identity failed')


def analyze(y, fixed, layers, predictions, groups, weights):
    """All groups of each family must partition the full window, including empty cells.

    Arrays are finite float64 and alignment is caller-owned. C is R102 zero C.
    Contributions use the full-window denominator/mass; subgroup means do not.
    """
    y = vector(y, None, 'y'); n = len(y); need(n > 0, 'Empty full window')
    w = vector(weights, n, 'weights'); need(np.all(w >= 0) and abs(w.sum()-1.) < 1e-12, 'Invalid global weights')
    need(set(fixed) == {'area', 'lgb', 'H', 'U'} and set(layers) == set(predictions) == set(ARMS), 'Unexpected roles')
    f = {k: vector(v, n, k) for k, v in fixed.items()}
    valid_groups = []; families = {}; seen = set()
    for family, name, mask in groups:
        need(isinstance(family, str) and isinstance(name, str) and (family, name) not in seen, 'Invalid/duplicate group')
        mask = np.asarray(mask); need(mask.dtype == np.bool_ and mask.shape == (n,), 'Invalid group mask')
        seen.add((family, name)); valid_groups.append((family, name, mask)); families.setdefault(family, []).append(mask)
    need(bool(families), 'At least one complete group family required')
    for family, masks in families.items(): need(np.all(np.sum(masks, axis=0) == 1), 'Group family is not a partition: ' + family)
    lo = np.minimum(f['H'], f['U']); hi = np.maximum(f['H'], f['U'])
    eA = f['area']-y; eL = f['lgb']-y; remainder = .375*eA+.25*eL
    errors = {}; cancellation = {}; clipping = {}; routes = {}; metrics = {}; sign_tables = {}; checks = {}
    for arm in ARMS:
        need(set(layers[arm]) == set(LAYERS[:-1]), 'Unexpected layer roles')
        a = {k: vector(v, n, arm+'/'+k) for k, v in layers[arm].items()}
        a['median'] = vector(predictions[arm], n, arm+'/median')
        base = .75*(.5*f['area']+.5*a['native'])+.25*f['lgb']
        median = np.median(np.column_stack([a['spatial'], f['H'], f['U']]), axis=1)
        clipped = np.clip(a['spatial'], lo, hi)
        need(np.array_equal(base, a['base']), 'Saved base formula differs: ' + arm)
        need(np.array_equal(median, a['median']) and np.array_equal(clipped, a['median']), 'Saved median formula differs: ' + arm)
        errors[arm] = {k: np.abs(p-y) for k, p in a.items()}
        cancellation[arm] = .375*np.abs(eA)+.375*errors[arm]['native']+.25*np.abs(eL)-errors[arm]['base']
        need(cancellation[arm].min() >= -TOL, 'Negative cancellation beyond roundoff')
        clipping[arm] = errors[arm]['median']-errors[arm]['spatial']; routes[arm] = route(a['spatial'], lo, hi)
        metrics[arm] = {k: {'overall_mae': float(e.mean()), 'weighted_mae': float(np.dot(w, e))} for k, e in errors[arm].items()}
        checks[arm] = {'base_formula_exact': True, 'median_and_clip_exact': True, 'minimum_K_TWD': float(cancellation[arm].min())}
        q = {'native_abs': errors[arm]['native'], 'fixed_abs': np.abs(remainder), 'base_abs': errors[arm]['base'], 'K': cancellation[arm]}
        pm = sign_masks(a['native']-y); rm = sign_masks(remainder)
        sign_tables[arm] = [{'native_sign': p, 'fixed_error_sign': r, **summarize(pm[p] & rm[r], q, w)} for p, r in itertools.product(SIGNS, repeat=2)]
        check_sum(sign_tables[arm], summarize(np.ones(n, dtype=bool), q, w), 'Native sign cells')
    comparisons = {}
    for control in ('v22', 'C'):
        q = {'native_term': .375*(errors['T']['native']-errors[control]['native']),
             'cancellation_term': -(cancellation['T']-cancellation[control]),
             'base_delta': errors['T']['base']-errors[control]['base'],
             'spatial_delta': errors['T']['spatial']-errors[control]['spatial'],
             'clipping_term': clipping['T']-clipping[control],
             'median_delta': errors['T']['median']-errors[control]['median']}
        br = float(np.max(np.abs(q['base_delta']-q['native_term']-q['cancellation_term'])))
        mr = float(np.max(np.abs(q['median_delta']-q['spatial_delta']-q['clipping_term'])))
        need(br <= TOL and mr <= TOL, 'Row algebra identity failed')
        whole = summarize(np.ones(n, dtype=bool), q, w)
        gs = [{'family': family, 'name': name, **summarize(mask, q, w)} for family, name, mask in valid_groups]
        rs = [{'control_state': old, 'T_state': new, **summarize(routes[control][old] & routes['T'][new], q, w)} for old, new in itertools.product(ROUTES, repeat=2)]
        sm = sign_masks(q['spatial_delta'], CHANGES); mm = sign_masks(q['median_delta'], CHANGES)
        es = [{'spatial_change': old, 'median_change': new, **summarize(sm[old] & mm[new], q, w)} for old, new in itertools.product(CHANGES, repeat=2)]
        for record in [whole, *gs, *rs, *es]: check_identity(record)
        for family in families: check_sum([g for g in gs if g['family'] == family], whole, 'Group ' + family)
        check_sum(rs, whole, 'Median routes'); check_sum(es, whole, 'Error change cells')
        comparisons['T-'+control] = {'overall': whole, 'groups': gs, 'median_routes': rs, 'error_sign_transitions': es,
            'maximum_base_identity_residual_TWD': br, 'maximum_median_identity_residual_TWD': mr}
    result = {'schema_version': 1, 'round': 103, 'status': 'descriptive_math_completed', 'rows': n,
              'parameters': parameters(), 'metrics': metrics, 'identity_checks': checks,
              'comparisons': comparisons, 'native_fixed_error_signs': sign_tables,
              'C_role': 'R102 same-width positive-zero control, not R100 C',
              'statistical_or_causal_inference': False, 'new_fit_or_bootstrap': False}
    json.dumps(result, allow_nan=False)
    return result


def self_check():
    tick = time.monotonic(); checks = {}
    def mark(name, ok): need(bool(ok), name); checks[name] = True
    def rejects(name, fn):
        try: fn()
        except ValueError: checks[name] = True
        else: raise AssertionError(name)
    y = np.full(8, 100., dtype=np.float64)
    fixed = {'area': y-20, 'lgb': y-20, 'H': np.array([90.]*5+[100.]*3), 'U': np.array([110.]*5+[100.]*3)}
    ns = {'v22': np.array([80., 90., 100., 110., 120., 100., 80., 120.]),
          'C': np.array([95., 100., 90., 110., 120., 100., 105., 90.]),
          'T': np.array([95., 115., 85., 100., 110., 105., 100., 90.])}
    layers = {}; predictions = {}
    for arm, delta in zip(ARMS, (40., 30., 10.)):
        native = y+delta
        layers[arm] = {'native': native, 'base': .75*(.5*fixed['area']+.5*native)+.25*fixed['lgb'], 'reference': ns[arm]+1., 'spatial': ns[arm]}
        predictions[arm] = np.median(np.column_stack([ns[arm], fixed['H'], fixed['U']]), axis=1)
    w = np.array([.1, .15, .2, .1, .05, .4, 0., 0.]); allmask = np.ones(8, dtype=bool)
    groups = [('all', 'all', allmask), ('test', 'positive_mass', w > 0), ('test', 'zero_mass', w == 0), ('test', 'empty', ~allmask)]
    before = json.dumps({k: v.tolist() for k, v in fixed.items()}, sort_keys=True)
    r = analyze(y, fixed, layers, predictions, groups, w); d = r['comparisons']['T-v22']; q = d['overall']['quantities']
    mark('native_improves_but_base_worsens', q['native_term']['ordinary_mean'] == -11.25 and q['base_delta']['ordinary_mean'] == 6.25)
    mark('cancellation_loss_exactly_explains_reversal', q['cancellation_term']['ordinary_mean'] == 17.5)
    mark('both_row_identities', d['maximum_base_identity_residual_TWD'] == d['maximum_median_identity_residual_TWD'] == 0.)
    mark('all_36_routes_retained', len(d['median_routes']) == 36 and sum(v['rows'] for v in d['median_routes']) == 8)
    mark('all_six_route_labels_present', {v['control_state'] for v in d['median_routes'] if v['rows']} == set(ROUTES))
    tied = route(ns['v22'][5:], fixed['H'][5:], fixed['U'][5:])
    mark('equal_bounds_below_equal_above', tied['both_equal'].tolist() == [True, False, False] and tied['below'].tolist() == [False, True, False] and tied['above'].tolist() == [False, False, True])
    mark('nine_error_change_cells_retained', len(d['error_sign_transitions']) == 9)
    mark('nine_native_error_cells_per_arm', all(len(c) == 9 for c in r['native_fixed_error_signs'].values()))
    empty = next(g for g in d['groups'] if g['name'] == 'empty'); zero = next(g for g in d['groups'] if g['name'] == 'zero_mass')
    mark('empty_group_null_means_zero_contributions', empty['rows'] == 0 and all(v['ordinary_mean'] is None and v['weighted_mean'] is None and v['ordinary_contribution'] == v['weighted_contribution'] == 0 for v in empty['quantities'].values()))
    mark('zero_mass_preserves_ordinary_mean', zero['rows'] == 2 and all(v['ordinary_mean'] is not None and v['weighted_mean'] is None and v['weighted_contribution'] == 0 for v in zero['quantities'].values()))
    cell = next(v for v in d['median_routes'] if v['control_state'] == 'below' and v['T_state'] == 'inside')
    mark('weighted_mean_is_not_contribution', cell['weight_mass'] == .1 and cell['quantities']['base_delta']['weighted_mean'] == 6.25 and cell['quantities']['base_delta']['weighted_contribution'] == .625)
    mark('all_five_layer_metrics', all(set(v) == set(LAYERS) for v in r['metrics'].values()))
    mark('inputs_unchanged', before == json.dumps({k: v.tolist() for k, v in fixed.items()}, sort_keys=True))
    perm = np.array([7, 3, 0, 6, 1, 4, 2, 5]); permuted = analyze(y[perm], {k:v[perm] for k,v in fixed.items()}, {a:{k:v[perm] for k,v in ls.items()} for a,ls in layers.items()}, {a:v[perm] for a,v in predictions.items()}, [(f,n,m[perm]) for f,n,m in groups], w[perm])
    mark('joint_row_permutation_metrics', all(abs(r['metrics'][a][l][m]-permuted['metrics'][a][l][m]) < 1e-12 for a in ARMS for l in LAYERS for m in ('overall_mae','weighted_mae')))
    rejects('bad_weights_rejected', lambda: analyze(y, fixed, layers, predictions, groups, w*.5))
    rejects('nonpartition_groups_rejected', lambda: analyze(y, fixed, layers, predictions, [('bad','one',allmask),('bad','two',allmask)], w))
    bad = {a:{**v} for a,v in layers.items()}; bad['T']['base'] = bad['T']['base']+1.
    rejects('formula_source_mismatch_rejected', lambda: analyze(y, fixed, bad, predictions, groups, w))
    badp = {a:v.copy() for a,v in predictions.items()}; badp['T'][0] += 1.
    rejects('median_source_mismatch_rejected', lambda: analyze(y, fixed, layers, badp, groups, w))
    mark('strict_json_serializable', isinstance(json.loads(json.dumps(r, allow_nan=False)), dict))
    return {'status': 'passed', 'exit_code': 0, 'check_count': len(checks), 'checks': checks,
            'source_sha256': source_hashes(), 'plan_sha256': PLAN_SHA, 'parameters': parameters(),
            'synthetic_only': True, 'real_data_read': False, 'new_fit_predict_score_bootstrap_calls': 0,
            'elapsed_seconds': time.monotonic()-tick, 'completed_at_utc': datetime.now(timezone.utc).isoformat()}


def write_receipt(path, value):
    path = Path(path); side = path.with_suffix('.sha256')
    need(not path.exists() and not side.exists(), 'Preserve existing receipt')
    with path.open('x', encoding='utf8', newline='\n') as f: json.dump(value, f, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False); f.write('\n')
    with side.open('x', encoding='ascii') as f: f.write(hashlib.sha256(path.read_bytes()).hexdigest()+'\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--self-check', action='store_true'); parser.add_argument('--receipt', default=str(ROOT/'reports/round103_math_selfcheck.json')); args = parser.parse_args()
    if not args.self_check: parser.error('--self-check is required; no real-data CLI exists')
    try: receipt = self_check()
    except Exception:
        write_receipt(args.receipt, {'status':'failed','exit_code':1,'failure':traceback.format_exc(),'source_sha256':source_hashes(),'synthetic_only':True,'real_data_read':False}); raise
    write_receipt(args.receipt, receipt); print(json.dumps({'status':receipt['status'],'checks':receipt['check_count'],'elapsed_seconds':receipt['elapsed_seconds']}))
