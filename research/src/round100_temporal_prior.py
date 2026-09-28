"""Quarter-prefix, group-held-out fit priors for the fixed R100 experiment.

No query labels, query features, NN fitting, or file access occurs here.
Calendar/fold masks reuse the independently saved pure-X policy. R60 builds
each price state only from the lawful reference; excluded warmup rows have
no fabricated training factor. A caller may preserve states with state_sink.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import time
import numpy as np
import pandas as pd
import round60_recency_price_prior as prior
import round100_temporal_prior_x_audit as audit
import round100_contract as c


def content_sha(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
        separators=(',', ':'), allow_nan=False).encode('utf8')).hexdigest()


def build_temporal_train_prior(Xfit, yfit, canonical_indices, expected_folds,
                               expected_eligible, *, state_sink=None):
    """Return (positive float64 priors, int64 kept positions, JSON evidence).

    state_sink(start_day, held_fold, state) optionally persists an immutable
    JSON state and returns its file record. Reference masks always exclude
    invalid dates, the current/later quarter, and the entire original fold.
    """
    tick = time.monotonic()
    n = len(Xfit)
    ix = np.asarray(canonical_indices)
    c.need(n > 0 and ix.dtype == np.int64 and ix.shape == (n,)
           and (ix >= 0).all() and len(np.unique(ix)) == n, 'Canonical int64 indices required')
    if isinstance(yfit, pd.Series):
        c.need(yfit.index.equals(Xfit.index), 'Fit Series order differs')
    y = np.asarray(yfit)
    c.need(y.dtype == np.float64 and y.shape == (n,) and np.isfinite(y).all()
           and (y > 0).all(), 'Full-precision positive float64 fit labels required')
    c.need(isinstance(expected_folds, np.ndarray) and expected_folds.dtype == np.int64
           and expected_folds.shape == (n,), 'Expected int64 folds required')
    c.need(isinstance(expected_eligible, np.ndarray) and expected_eligible.dtype == bool
           and expected_eligible.shape == (n,), 'Expected Boolean eligibility required')
    keys = prior.original._keys(Xfit)
    dates = prior._dates(Xfit)
    days, invalid, outside = dates
    valid = np.isfinite(days) & (invalid == 0) & (outside == 0)
    folds, groups, fallback = audit.folds_for(Xfit, ix)
    c.exact(folds, expected_folds, 'Original fold assignment differs')
    qstart, eligible, quarters = audit.policy(days, valid, folds)
    c.exact(eligible, expected_eligible, 'Frozen pure-X eligible mask differs')
    c.need(eligible.any(), 'No eligible training rows')
    kept = np.flatnonzero(eligible).astype(np.int64)
    position = np.full(n, -1, dtype=np.int64)
    position[kept] = np.arange(len(kept), dtype=np.int64)
    area = prior.original.effective_area(Xfit)
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        units = y / area
    c.need(np.isfinite(units).all() and (units > 0).all(), 'Invalid unit prices')
    result = np.full(len(kept), np.nan, dtype=np.float64)
    counts = {**{'|'.join(cols): 0 for cols in prior.LEVELS}, 'global': 0}
    records = []
    for quarter in quarters:
        if not quarter['eligible_quarter']:
            continue
        start = quarter['quarter_start_day']
        for fold in range(5):
            held = eligible & (qstart == start) & (folds == fold)
            reference = valid & (days < start) & (folds != fold)
            c.need(int(reference.sum()) >= 30, 'Insufficient frozen prefix support')
            if not held.any():
                continue
            c.need(not (held & reference).any() and days[reference].max() < start,
                   'Held/current/future reference row')
            c.need(not set(groups[reference]).intersection(groups[held]), 'Held group entered reference')
            state = prior._build(keys, units, dates, reference, 'quarter_prefix_for_held_fold')
            predicted, level_counts = prior._predict(state, [k[held] for k in keys], True)
            restored = json.loads(json.dumps(state, ensure_ascii=False, allow_nan=False))
            c.exact(predicted, prior._predict(restored, [k[held] for k in keys]), 'Prior JSON roundtrip')
            result[position[np.flatnonzero(held)]] = predicted
            for name, value in level_counts.items():
                counts[name] += value
            saved = state_sink(int(start), fold, state) if state_sink is not None else None
            records.append({'quarter': quarter['quarter'], 'quarter_start_day': int(start),
                'held_fold': fold, 'held_rows': int(held.sum()), 'reference_rows': int(reference.sum()),
                'reference_groups': len(np.unique(groups[reference])),
                'reference_min_day': float(days[reference].min()),
                'reference_max_day': float(days[reference].max()),
                'reference_indices': c.identity(ix[reference]), 'held_indices': c.identity(ix[held]),
                'reference_target': c.identity(y[reference]), 'reference_units': c.identity(units[reference]),
                'reference_dates': [c.identity(a[reference]) for a in dates],
                'state_content_sha256': content_sha(state), 'saved_state': saved,
                'global_median': state['global_median'], 'date_support': state['date_support'],
                'level_support': [{k: level[k] for k in ('columns', 'cell_count', 'minimum_observed_support')}
                                  for level in state['levels']],
                'held_prior_level_counts': level_counts, 'held_prior': c.identity(predicted),
                'JSON_reload_exact': True})
    c.need(np.isfinite(result).all() and (result > 0).all() and sum(counts.values()) == len(kept),
           'Unassigned/nonpositive prior or support count mismatch')
    evidence = {'status': 'quarter_prefix_priors_completed', 'fit_rows': n, 'eligible_rows': len(kept),
        'warmup_rows': int((valid & ~eligible).sum()), 'invalid_date_rows': int((~valid).sum()),
        'prefix_state_count': len(records), 'prefix_states_saved': state_sink is not None,
        'saved_state_records_complete': state_sink is not None and all(r['saved_state'] is not None for r in records),
        'rules': audit.rules(), 'price_rules': prior.RULES, 'quarters': quarters, 'prefixes': records,
        'held_prior_level_counts': counts, 'fold_ids': c.identity(folds), 'eligible_mask': c.identity(eligible),
        'kept_positions': c.identity(kept), 'canonical_indices': c.identity(ix),
        'fit_target': c.identity(y), 'effective_area': c.identity(area), 'fit_units': c.identity(units),
        'temporal_prior': c.identity(result), 'group_keys_sha256': content_sha(groups.tolist()),
        'fallback_counts': fallback, 'same_quarter_reference_allowed': False,
        'later_quarter_reference_allowed': False, 'held_group_reference_allowed': False,
        'uses_query_labels': False, 'uses_query_features': False, 'full_query_prior_rebuilt': False,
        'chronological_fit_prior_only_not_full_encoder_or_NN': True,
        'elapsed_seconds': time.monotonic() - tick}
    return result, kept, evidence


def self_check():
    tick = time.monotonic(); checks = []
    n = 480
    X = pd.DataFrame({'project_name': [f'p{i % 80}' for i in range(n)], 'address_city': ['A'] * n,
        'address_district': ['D'] * n, 'address_raw': ['__MISSING__'] * n,
        'address_road': ['__MISSING__'] * n, 'address_section': ['__MISSING__'] * n,
        'transaction_type': ['house'] * n, 'building_type': ['apartment'] * n,
        'building_area': np.full(n, 100., dtype=np.float32), 'parking_area': np.zeros(n, dtype=np.float32),
        'trade_day_index': np.repeat([10., 91., 182., 274.], 120),
        'trade_date_invalid': np.zeros(n), 'trade_date_out_of_range': np.zeros(n)})
    # Year 2000 is leap: April 1 == 91, July 1 == 182, October 1 == 274.
    indices = np.arange(4000, 4000+n, dtype=np.int64)
    y = np.linspace(100., 1000., n, dtype=np.float64) * 100
    before = X.copy(deep=True); ybefore = y.copy()
    folds, _, _ = audit.folds_for(X, indices)
    days = X.trade_day_index.to_numpy(); valid = np.ones(n, dtype=bool)
    q, mask, _ = audit.policy(days, valid, folds)
    states = {}
    def sink(start, fold, state):
        key = (start, fold); c.need(key not in states, 'Duplicate state')
        states[key] = json.loads(json.dumps(state))
        return {'synthetic_key': list(key), 'content_sha256': content_sha(state)}
    result, kept, evidence = build_temporal_train_prior(X, y, indices, folds, mask, state_sink=sink)
    c.need(np.array_equal(kept, np.arange(120, n)) and set(q) == {0, 91, 182, 274}, 'Quarter boundary or warmup')
    checks += ['calendar_boundaries', 'warmup_omitted_not_filled', 'partial_last_quarter_retained']
    c.need(evidence['prefix_state_count'] == 15 and evidence['saved_state_records_complete'], 'All states saved')
    checks += ['state_callback_once_per_nonempty_prefix', 'state_JSON_roundtrip']
    target = (q[kept] == 182) & (folds[kept] == 0)
    changed_y = y.copy(); changed_y[(days >= 182) | (folds == 0)] *= 100
    revised, _, ev2 = build_temporal_train_prior(X, changed_y, indices, folds, mask)
    c.exact(result[target], revised[target], 'Same/future/held labels leaked')
    r1 = next(r for r in evidence['prefixes'] if r['quarter_start_day'] == 182 and r['held_fold'] == 0)
    r2 = next(r for r in ev2['prefixes'] if r['quarter_start_day'] == 182 and r['held_fold'] == 0)
    c.need(r1['state_content_sha256'] == r2['state_content_sha256'], 'Reference state changed')
    checks += ['same_quarter_label_invariance', 'future_label_invariance', 'held_fold_label_invariance', 'reference_state_invariance']
    keys = prior.original._keys(X)
    ref = (days < 182) & (folds != 0)
    direct = prior._build(keys, y/100, prior._dates(X), ref, 'quarter_prefix_for_held_fold')
    c.need(content_sha(direct) == content_sha(states[(182, 0)]), 'Original R60 estimator differs')
    checks.append('original_R60_estimator_exact')
    # Reverse fit order and reverse canonical identities together: keep lawful correspondence.
    reverse = np.arange(n-1, -1, -1)
    xr = X.iloc[reverse].copy(); fr, _, _ = audit.folds_for(xr, indices[reverse])
    mr = mask[reverse].copy()
    rr, kr, _ = build_temporal_train_prior(xr, y[reverse], indices[reverse], fr, mr)
    c.need(np.allclose(rr, result[::-1], rtol=1e-14, atol=0), 'Row permutation correspondence')
    checks.append('row_order_correspondence_tolerance')
    invalid_X = X.copy(); invalid_X.loc[0, 'trade_day_index'] = np.nan
    invalid_X.loc[0, 'trade_date_invalid'] = 1
    d, bad, out = prior._dates(invalid_X); vv = np.isfinite(d) & (bad == 0) & (out == 0)
    _, mm, _ = audit.policy(d, vv, folds)
    _, _, ie = build_temporal_train_prior(invalid_X, y, indices, folds, mm)
    c.need(ie['invalid_date_rows'] == 1 and all(r['reference_rows'] < 120 if r['quarter_start_day'] == 91 else True for r in ie['prefixes']), 'Invalid dates entered prefix')
    checks.append('invalid_date_excluded_from_prior_and_loss')
    badflag = X.copy(); badflag.loc[0, 'trade_date_invalid'] = np.nan
    for name, xx, yy, ii, ff, mm in [
        ('wrong_mask', X, y, indices, folds, ~mask),
        ('wrong_fold', X, y, indices, (folds + 1) % 5, mask),
        ('duplicate_index', X, y, np.zeros(n, dtype=np.int64), folds, mask),
        ('float32_y', X, y.astype(np.float32), indices, folds, mask),
        ('nonpositive_y', X, y*0, indices, folds, mask),
        ('nonfinite_flag', badflag, y, indices, folds, mask)]:
        try: build_temporal_train_prior(xx, yy, ii, ff, mm)
        except ValueError: checks.append('reject_' + name)
        else: raise AssertionError('Invalid synthetic case accepted: ' + name)
    pd.testing.assert_frame_equal(X, before, check_exact=True); c.exact(y, ybefore, 'Caller target mutated')
    checks.append('caller_inputs_unchanged')
    return {'status': 'passed', 'check_count': len(checks), 'checks': checks, 'real_data_read': False,
            'new_real_prior_builds': 0, 'new_native_fits': 0, 'GPU_calls': 0,
            'source_sha256': c.sha(__file__), 'elapsed_seconds': time.monotonic()-tick}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--self-check', action='store_true', required=True)
    parser.add_argument('--receipt')
    args = parser.parse_args(); result = self_check()
    if args.receipt: c.write(args.receipt, result)
    print(json.dumps(result, ensure_ascii=False))
