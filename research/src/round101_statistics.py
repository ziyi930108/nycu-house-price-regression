"""Fixed descriptive statistics for R101; no data access, model, or score."""
from __future__ import annotations
import time
import numpy as np

QUANTILES = ('0', '0.01', '0.1', '0.25', '0.5', '0.75', '0.9', '0.99', '1')
PROBABILITIES = np.array([float(q) for q in QUANTILES], dtype=np.float64)


def need(ok, message):
    if not ok:
        raise ValueError(message)


def describe(values, weights=None, reference_weight_sum=None):
    """Describe every supplied row; never silently drop nonfinite values.

    Explicit weights use inverse-CDF quantiles, with exact endpoint min/max
    (positive mathematical mass); unweighted quantiles use NumPy linear.
    The denominator for a partition's mass is supplied from its full scope.
    """
    x = np.asarray(values, dtype=np.float64)
    need(x.ndim == 1 and np.isfinite(x).all(), 'Finite one-dimensional values required')
    n = len(x)
    weighted = weights is not None
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    need(w.shape == x.shape and np.isfinite(w).all() and (w > 0).all(), 'Positive aligned weights required')
    total = float(w.sum(dtype=np.float64))
    need(np.isfinite(total), 'Weight sum overflow')
    if reference_weight_sum is None:
        denominator = total
    else:
        denominator = float(reference_weight_sum)
        need(np.isfinite(denominator) and denominator >= 0
             and total <= denominator + 1e-12 * max(1., denominator), 'Invalid reference weight mass')
    result = {'rows': n, 'weight_sum': total,
              'weight_mass_fraction': total/denominator if denominator > 0 else None,
              'ESS': 0. if not n else None, 'mean': None, 'std': None, 'min': None, 'max': None,
              'quantile_method': 'inverse_CDF_positive_mass_endpoint_minmax' if weighted else 'numpy_linear',
              'quantiles': {q: None for q in QUANTILES}}
    if not n:
        return result
    need(total > 0, 'Nonempty distribution requires positive mass')
    normalized = w / total
    mean = float(np.dot(normalized, x))
    variance = float(np.dot(normalized, np.square(x-mean)))
    need(np.isfinite(mean) and np.isfinite(variance), 'Moment overflow')
    if weighted:
        order = np.argsort(x, kind='stable')
        ordered_x = x[order]
        cumulative = np.cumsum(w[order], dtype=np.float64)
        positions = np.searchsorted(cumulative, PROBABILITIES*cumulative[-1], side='left')
        positions = np.minimum(positions, n-1)
        positions[0], positions[-1] = 0, n-1
        quantiles = ordered_x[positions]
    else:
        quantiles = np.quantile(x, PROBABILITIES, method='linear')
    result.update(ESS=float(1./np.dot(normalized, normalized)), mean=mean,
                  std=float(np.sqrt(variance)), min=float(x.min()), max=float(x.max()),
                  quantiles={q: float(value) for q, value in zip(QUANTILES, quantiles)})
    return result


def self_check():
    tick = time.monotonic(); checks = []
    def mark(name, ok):
        need(ok, name); checks.append(name)
    x = np.array([1., 3., 5., 9.]); w = np.array([1., 1., 1., 5.])
    before = x.copy(), w.copy()
    ordinary = describe(x)
    weighted = describe(x, w)
    mark('linear_median', ordinary['quantiles']['0.5'] == 4.)
    mark('ordinary_mean_population_std', ordinary['mean'] == 4.5 and np.isclose(ordinary['std'], np.std(x, ddof=0)))
    mark('inverse_CDF_median', weighted['quantiles']['0.5'] == 9.)
    mark('inverse_CDF_exact_half_not_midpoint', describe([1., 3.], [1., 1.])['quantiles']['0.5'] == 1.)
    mark('weighted_mean_ESS', weighted['mean'] == 6.75 and np.isclose(weighted['ESS'], 64./28.))
    part = describe(x[:2], w[:2], reference_weight_sum=8.)
    mark('partition_mass_uses_full_scope', part['weight_mass_fraction'] == .25)
    empty = describe([], [], reference_weight_sum=8.)
    mark('empty_group_retained', empty['rows'] == 0 and empty['weight_mass_fraction'] == 0 and all(v is None for v in empty['quantiles'].values()))
    mark('empty_full_scope_null_mass', describe([])['weight_mass_fraction'] is None)
    translated = describe(x-20, w)
    mark('signed_log_values_supported', np.isclose(translated['mean'], weighted['mean']-20) and np.isclose(translated['std'], weighted['std']))
    scaled = describe(x, w*19.)
    mark('common_weight_scale_invariant', scaled['quantiles'] == weighted['quantiles'] and np.isclose(scaled['mean'], weighted['mean']) and np.isclose(scaled['ESS'], weighted['ESS']))
    permuted = describe(x[::-1], w[::-1])
    mark('row_permutation_invariant', permuted['quantiles'] == weighted['quantiles'] and np.isclose(permuted['mean'], weighted['mean']))
    tail = describe([1., 9.], [1., 1e-30])
    mark('positive_mass_endpoint_not_lost_to_roundoff', tail['quantiles']['1'] == tail['max'] == 9.)
    for name, args in [('nan', ([1., np.nan],)), ('zero_weight', ([1., 2.], [1., 0.])),
                       ('negative_weight', ([1., 2.], [1., -1.])), ('misaligned', ([1., 2.], [1.])),
                       ('matrix', (np.ones((2, 2)),)), ('invalid_scope', ([1., 2.], [1., 1.], 1.))]:
        try: describe(*args)
        except ValueError: checks.append('reject_' + name)
        else: raise AssertionError('Invalid input accepted: ' + name)
    mark('caller_arrays_unchanged', np.array_equal(x, before[0]) and np.array_equal(w, before[1]))
    # Fixed positive factors: algebraic L1 equivalence, not float32 bitwise claim.
    y = np.array([2., 5., 12.]); factor = np.array([.5, 2., 4.]); ratio = np.array([3., 1., 2.])
    weights = np.array([.2, 1., 1.8])
    mark('positive_factor_ratio_loss_algebra', np.allclose(weights*np.abs(factor*ratio-y), weights*factor*np.abs(ratio-y/factor), rtol=1e-14, atol=1e-14))
    return {'status': 'passed', 'check_count': len(checks), 'checks': checks,
            'real_data_read': False, 'fit_predict_score_calls': 0, 'elapsed_seconds': time.monotonic()-tick}
