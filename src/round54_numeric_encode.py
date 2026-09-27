"""R54 fit-only quantile ramps appended to unchanged R48 144 numeric inputs."""
from __future__ import annotations
import json
import numpy as np

INPUT_WIDTH=144
NUMERIC_COLUMNS=72
QUANTILES=np.linspace(0.0,1.0,17)


def _input(values):
    array=np.asarray(values)
    if array.dtype!=np.float32 or array.ndim!=2 or array.shape[1]!=INPUT_WIDTH:
        raise ValueError('Expected float32 (rows,144) numeric inputs in their original order')
    if not np.isfinite(array).all():
        raise ValueError('Numeric inputs must be finite; no new missing-value rule')
    return array


def fit_state(train_num):
    """Learn only from train_num; first 72 numeric columns, no labels/query."""
    values=_input(train_num)
    if len(values)==0:raise ValueError('Cannot fit quantiles without training rows')
    quantiles=np.quantile(values[:,:NUMERIC_COLUMNS],QUANTILES,axis=0,method='linear')
    columns=[];offset=INPUT_WIDTH
    for column in range(NUMERIC_COLUMNS):
        knots=np.unique(quantiles[:,column]).astype(np.float64)
        width=len(knots)-1
        columns.append({'column':column,'knots':knots.tolist(),'offset':offset,'width':width})
        offset+=width
    return {'version':1,'input_width':INPUT_WIDTH,'numeric_columns':NUMERIC_COLUMNS,
            'quantiles':QUANTILES.tolist(),'quantile_method':'linear','fit_rows':len(values),
            'columns':columns,'output_width':offset,'ramp_width':offset-INPUT_WIDTH,
            'fit_uses_target':False,'fit_uses_query':False,'original_144_preserved':True,
            'ramp_formula':'clip((x-left)/(right-left),0,1)','ramp_standardization':False,
            'output_dtype':'float32'}


def transform(state,num):
    """Apply frozen knots in order; out-of-range ramps saturate, original values remain."""
    values=_input(num)
    if (state['version']!=1 or state['input_width']!=INPUT_WIDTH or state['numeric_columns']!=NUMERIC_COLUMNS
        or state['quantiles']!=QUANTILES.tolist() or state['quantile_method']!='linear'
        or len(state['columns'])!=NUMERIC_COLUMNS or state['fit_rows']<=0
        or state['fit_uses_target'] is not False or state['fit_uses_query'] is not False
        or state['output_dtype']!='float32' or state['ramp_standardization'] is not False):
        raise ValueError('Frozen encoder contract differs')
    offset=INPUT_WIDTH;validated=[]
    for column,entry in enumerate(state['columns']):
        knots=np.asarray(entry['knots'],dtype=np.float64)
        if (knots.ndim!=1 or not 1<=len(knots)<=17 or not np.isfinite(knots).all()
            or not (np.diff(knots)>0).all() or entry['column']!=column
            or entry['offset']!=offset or entry['width']!=len(knots)-1):
            raise ValueError('Invalid frozen knots/column offsets')
        validated.append(knots);offset+=len(knots)-1
    if state['output_width']!=offset or state['ramp_width']!=offset-INPUT_WIDTH:
        raise ValueError('Frozen output width differs')
    result=np.empty((len(values),offset),dtype=np.float32)
    result[:,:INPUT_WIDTH]=values
    for entry,knots in zip(state['columns'],validated):
        width=entry['width']
        if width:
            x=values[:,entry['column']].astype(np.float64)[:,None]
            ramps=np.clip((x-knots[:-1])/(knots[1:]-knots[:-1]),0.0,1.0)
            result[:,entry['offset']:entry['offset']+width]=ramps.astype(np.float32)
    if not np.isfinite(result).all():raise ValueError('Nonfinite encoded values')
    return result


def self_check():
    train=np.zeros((17,144),dtype=np.float32)
    train[:,0]=np.arange(17,dtype=np.float32)
    train[:,1]=np.r_[np.zeros(8),np.ones(9)].astype(np.float32)
    train[:,2]=5.0;train[:,3]=-0.0
    train[:,72:]=np.arange(72,dtype=np.float32)%2
    before=train.tobytes();state=fit_state(train)
    assert state['columns'][0]['width']==16 and state['columns'][1]['width']==1
    assert state['columns'][2]['width']==0 and state['output_width']==161
    query=train[[16,0,8]].copy();query[0,0]=100;query[1,0]=-100;query[0,1]=2;query[1,1]=-1
    qbefore=query.tobytes();result=transform(state,query)
    assert result.dtype==np.float32 and result.shape==(3,161)
    assert result[:,:144].tobytes()==qbefore and train.tobytes()==before and query.tobytes()==qbefore
    assert np.array_equal(result[0,144:160],np.ones(16,dtype=np.float32))
    assert np.array_equal(result[1,144:160],np.zeros(16,dtype=np.float32))
    assert np.array_equal(result[2,144:160],np.r_[np.ones(8),np.zeros(8)].astype(np.float32))
    assert np.array_equal(result[:,160],np.array([1,0,1],dtype=np.float32))
    restored=json.loads(json.dumps(state,allow_nan=False))
    assert transform(restored,query).tobytes()==result.tobytes()
    assert transform(state,query[::-1]).tobytes()==result[::-1].tobytes()
    constant=np.full((3,144),7,dtype=np.float32);constant_state=fit_state(constant)
    assert constant_state['ramp_width']==0 and transform(constant_state,constant).tobytes()==constant.tobytes()
    for invalid in (np.nan,np.inf,-np.inf):
        bad=query.copy();bad[0,0]=invalid
        for call in (lambda:fit_state(bad),lambda:transform(state,bad)):
            try:call()
            except ValueError:pass
            else:raise AssertionError('Nonfinite input was not rejected')
    return {'status':'passed','fit_rows':17,'output_width':state['output_width'],
            'constant_and_duplicate_quantiles':True,'out_of_range_saturation':True,
            'nonfinite_rejected':True,'JSON_roundtrip_exact':True,'row_order_preserved':True,
            'original_144_bitwise_preserved':True,'inputs_unchanged':True,'real_data_read':False,'GPU_calls':0}


if __name__=='__main__':print(json.dumps(self_check(),ensure_ascii=False))
