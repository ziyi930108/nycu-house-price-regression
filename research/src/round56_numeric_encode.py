"""R56: R54 quantile ramps excluding three named date columns, originals retained."""
from __future__ import annotations
import json
import numpy as np
import round54_numeric_encode as original

INPUT_WIDTH=original.INPUT_WIDTH
NUMERIC_COLUMNS=original.NUMERIC_COLUMNS
EXCLUDED_NAMES=('trade_year','trade_month_index','trade_day_index')


def _names(numeric_columns):
    names=list(numeric_columns)
    if (len(names)!=NUMERIC_COLUMNS or any(not isinstance(n,str) or not n for n in names)
        or len(set(names))!=NUMERIC_COLUMNS or not set(EXCLUDED_NAMES).issubset(names)):
        raise ValueError('Require 72 unique numeric names including all three exact date names')
    return names,[names.index(n) for n in EXCLUDED_NAMES]


def fit_state(train_num,numeric_columns):
    """Fit original R54 knots using training inputs only; remove named date ramps."""
    names,excluded=_names(numeric_columns)
    state=original.fit_state(train_num)
    values=original._input(train_num)
    offset=INPUT_WIDTH
    for entry in state['columns']:
        if entry['column'] in excluded:
            entry['knots']=[float(np.quantile(values[:,entry['column']],0.5,method='linear'))]
            entry['width']=0
        entry['offset']=offset
        offset+=entry['width']
    state.update(encoder='round56_no_date_ramps',numeric_column_names=names,
        excluded_numeric_names=list(EXCLUDED_NAMES),excluded_numeric_indices=excluded,
        excluded_columns_original_values_preserved=True,excluded_knots='fit median; no appended ramp',
        output_width=offset,ramp_width=offset-INPUT_WIDTH)
    return state


def transform(state,num):
    """Use frozen names/knots; unchanged R54 transformation for all retained ramps."""
    _,excluded=_names(state['numeric_column_names'])
    if (state['encoder']!='round56_no_date_ramps' or state['excluded_numeric_names']!=list(EXCLUDED_NAMES)
        or state['excluded_numeric_indices']!=excluded
        or state['excluded_columns_original_values_preserved'] is not True
        or any(state['columns'][j]['width']!=0 or len(state['columns'][j]['knots'])!=1 for j in excluded)):
        raise ValueError('Frozen date exclusion contract differs')
    return original.transform(state,num)


def self_check():
    names=['numeric_'+str(i) for i in range(NUMERIC_COLUMNS)]
    for i,n in zip((7,20,68),EXCLUDED_NAMES):names[i]=n
    train=np.zeros((17,INPUT_WIDTH),dtype=np.float32)
    train[:,:72]=(np.arange(17)[:,None]*(1+np.arange(72)%5)+np.arange(72)[None,:])/7
    train[:,4]=np.r_[np.zeros(8),np.ones(9)];train[:,5]=3;train[:,6]=-0.0
    train[:,72:]=np.arange(72)%2
    before=train.tobytes();train.flags.writeable=False
    full=original.fit_state(train);state=fit_state(train,names)
    assert state['excluded_numeric_indices']==[7,20,68]
    keep=list(range(INPUT_WIDTH))
    for entry in full['columns']:
        if entry['column'] not in state['excluded_numeric_indices']:
            keep.extend(range(entry['offset'],entry['offset']+entry['width']))
            assert state['columns'][entry['column']]['knots']==entry['knots']
    query=train[[16,0,8]].copy();query[0,0]=1000;query[1,0]=-1000
    query[0,7]=12345;query[1,20]=-6789
    query_before=query.tobytes();query.flags.writeable=False
    actual=transform(state,query);expected=original.transform(full,query)[:,keep]
    assert actual.dtype==np.float32 and actual.tobytes()==expected.tobytes()
    assert actual[:,:INPUT_WIDTH].tobytes()==query_before
    assert transform(json.loads(json.dumps(state,allow_nan=False)),query).tobytes()==actual.tobytes()
    assert transform(state,query[::-1]).tobytes()==actual[::-1].tobytes()
    assert train.tobytes()==before and query.tobytes()==query_before
    for invalid in (names[:7]+['not_trade_year']+names[8:],names[:-1]+[names[0]]):
        try:fit_state(train,invalid)
        except ValueError:pass
        else:raise AssertionError('Missing/duplicate numeric name accepted')
    return {'status':'passed','fit_rows':len(train),'excluded_indices':state['excluded_numeric_indices'],
        'R54_width':full['output_width'],'R56_width':state['output_width'],
        'exact_R54_slice_removal':True,'original_144_and_dates_bitwise_preserved':True,
        'JSON_roundtrip_exact':True,'row_order_preserved':True,'inputs_unchanged':True,
        'missing_or_duplicate_names_rejected':True,'real_data_read':False,'GPU_calls':0}


if __name__=='__main__':print(json.dumps(self_check(),ensure_ascii=False))
