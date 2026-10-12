"""Query-only neural array contract; artificial arrays can be checked without Torch."""
import numpy as np

def validate_query_arrays(arrays, spec, rows=47073):
    if set(arrays) != {'query_num','query_cat','cardinalities','query_factor_million'}:
        raise ValueError('Only query features/cardinalities/conversion factor accepted')
    qn,qc,card,factor = [arrays[k] for k in ('query_num','query_cat','cardinalities','query_factor_million')]
    if qn.dtype != np.float32 or qc.dtype != np.int64 or card.dtype != np.int64 or factor.dtype != np.float32:
        raise ValueError('Frozen query dtypes differ')
    if qn.shape != (rows,spec['numeric_count']) or qc.shape != (rows,len(spec['cardinalities'])):
        raise ValueError('Frozen neural row/column axis differs')
    if factor.shape != (rows,) or card.shape != (len(spec['cardinalities']),) or card.tolist() != spec['cardinalities']:
        raise ValueError('Frozen conversion-factor/cardinality schema differs')
    if not np.isfinite(qn).all() or not np.isfinite(factor).all() or not (factor > 0).all():
        raise ValueError('Invalid query numeric values/factors')
    for i,k in enumerate(card):
        if not ((qc[:,i] >= 0) & (qc[:,i] < k)).all():
            raise ValueError('Out-of-range neural category code')
    return qn,qc,card,factor
