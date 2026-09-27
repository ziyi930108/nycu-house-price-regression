"""Pure reported parking-area-per-slot representation; no fitting or file IO."""
from __future__ import annotations

import numpy as np
import pandas as pd

FEATURE_NAME = 'parking_area_per_slot'
FEATURES = (FEATURE_NAME,)
RULES = {
    'version': 1,
    'input_columns': ['parking_area', 'parking_count'],
    'output_columns': list(FEATURES),
    'formula': 'parking_area / parking_count',
    'eligibility': 'finite parking_area > 0 and finite integer parking_count > 0',
    'otherwise': 'NaN',
    'output_dtype': 'float32',
    'division_dtype': 'float64',
    'original_columns_and_index_unchanged': True,
    'caps_or_extra_flags': False,
    'fit_or_query_statistics': False,
    'float32_overflow': 'raise ValueError; do not cap or silently change eligibility',
    'semantics': 'reported transferred area per parsed slot count; not verified net slot size',
}


def features_from_X(X: pd.DataFrame) -> pd.DataFrame:
    """Return one float32 column on the same index; read only the two inputs."""
    if not isinstance(X, pd.DataFrame) or not X.columns.is_unique:
        raise ValueError('Expected a DataFrame with unique column names')
    columns = RULES['input_columns']
    if not set(columns).issubset(X.columns):
        raise ValueError('Missing parking area or count')
    for name in columns:
        dtype = X[name].dtype
        if (not pd.api.types.is_numeric_dtype(dtype)
                or pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_complex_dtype(dtype)):
            raise ValueError('Parking inputs must be real numeric columns')
    area = X['parking_area'].to_numpy(dtype='float64', na_value=np.nan, copy=True)
    count = X['parking_count'].to_numpy(dtype='float64', na_value=np.nan, copy=True)
    valid = (np.isfinite(area) & (area > 0) & np.isfinite(count)
             & (count > 0) & (count == np.floor(count)))
    ratio = np.full(len(X), np.nan, dtype='float64')
    np.divide(area, count, out=ratio, where=valid)
    try:
        with np.errstate(over='raise', invalid='raise'):
            result = ratio.astype('float32')
    except FloatingPointError as exc:
        raise ValueError('Eligible ratio cannot be represented as finite float32') from exc
    if not np.isfinite(result[valid]).all():
        raise ValueError('Eligible ratio must remain finite after float32 conversion')
    return pd.DataFrame({FEATURE_NAME: result}, index=X.index.copy())


def self_check():
    """Small in-memory boundaries, reserved for a future formal preflight."""
    index = pd.Index([9, 4, 4, 7, 2, 8, 1, 5, 3, 0], name='original_row')
    X = pd.DataFrame({
        'parking_area': [60., 60., 0., -1., np.nan, np.inf, 10., 10., 10., 10.],
        'parking_count': [1., 2., 1., 1., 1., 1., 0., -1., 1.5, np.inf],
    }, index=index)
    before = X.copy(deep=True)
    result = features_from_X(X)
    assert list(result.columns) == [FEATURE_NAME] and result[FEATURE_NAME].dtype == np.dtype('float32')
    assert np.array_equal(result[FEATURE_NAME].iloc[:2], np.array([60., 30.], dtype='float32'))
    assert result[FEATURE_NAME].iloc[2:].isna().all()
    assert result.index.equals(index) and X.equals(before) and X.dtypes.equals(before.dtypes)
    parts = pd.concat([features_from_X(X.iloc[:3]), features_from_X(X.iloc[3:])])
    assert result.equals(parts)
    assert features_from_X(X.iloc[:0]).shape == (0, 1)
    return {'synthetic_only': True, 'files_read': False, 'checks': [
        'single_and_multiple_slots', 'invalid_values_missing', 'float32_schema',
        'index_and_input_preserved', 'batch_invariance', 'empty_frame',
    ], 'check_count': 6}
