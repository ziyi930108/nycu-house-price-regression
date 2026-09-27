"""R75 descriptive review of two saved spatial predictions projected to fixed bounds.

No fitting, prediction, threshold search, new clipping rule, or causal claim.
The supplied final vectors must already be exactly the corresponding projections.
"""
from pathlib import Path
import argparse
import hashlib
import json
import time
import numpy as np

REGIONS=('below','within','above')
TRANSITIONS=tuple(f'{a}->{b}' for a in REGIONS for b in REGIONS)+('fixed_equal_bounds',)
SIGN_GROUPS=('improved','unchanged','worsened')


def projection_review(y,control_spatial,candidate_spatial,lower,upper,control_final,candidate_final):
    arrays=[np.asarray(v) for v in (y,control_spatial,candidate_spatial,lower,upper,control_final,candidate_final)]
    if any(a.dtype!=np.dtype('float64') or a.ndim!=1 for a in arrays):
        raise ValueError('All seven inputs must be one-dimensional float64 arrays')
    n=len(arrays[0])
    if n==0 or any(a.shape!=(n,) or not np.isfinite(a).all() for a in arrays):
        raise ValueError('Inputs must have equal nonempty lengths and finite values')
    y,s0,s1,lo,hi,f0,f1=arrays
    if not (lo<=hi).all():raise ValueError('lower must not exceed upper')
    if f0.tobytes()!=np.clip(s0,lo,hi).tobytes() or f1.tobytes()!=np.clip(s1,lo,hi).tobytes():
        raise ValueError('Saved final vectors must equal np.clip(spatial,lower,upper) bitwise')
    with np.errstate(over='raise',invalid='raise'):
        shift_s=s1-s0;shift_f=f1-f0
        errors={'spatial':(np.abs(s0-y),np.abs(s1-y)), 'final':(np.abs(f0-y),np.abs(f1-y))}
        deltas={k:b-a for k,(a,b) in errors.items()}
        transfer=deltas['final']-deltas['spatial']
    derived=[shift_s,shift_f,transfer,*deltas.values(),*(v for pair in errors.values() for v in pair)]
    if any(not np.isfinite(a).all() for a in derived):raise ValueError('Nonfinite derived differences')
    # Floating-point allowance concerns prediction changes only, not MAE gains.
    scale=np.maximum.reduce([np.ones(n),np.abs(s0),np.abs(s1),np.abs(lo),np.abs(hi),np.abs(f0),np.abs(f1)])
    tolerance=8*np.finfo(np.float64).eps*scale
    excess=np.abs(shift_f)-np.abs(shift_s)
    if not (excess<=tolerance).all():raise ValueError('Projection prediction-change contraction failed')
    def total(a):
        with np.errstate(over='raise',invalid='raise'):v=float(np.sum(a,dtype=np.float64))
        if not np.isfinite(v):raise ValueError('Nonfinite aggregate')
        return v
    def near(a,b,absolute_sum):
        return abs(a-b)<=64*np.finfo(np.float64).eps*max(1.0,absolute_sum)
    def shift_summary(values,mask):
        v=values[mask];count=len(v)
        return {'changed_rows':int(np.count_nonzero(v)), 'unchanged_rows':int(np.count_nonzero(v==0)),
                'mean_signed_TWD':None if not count else total(v)/count,
                'mean_absolute_TWD':None if not count else total(np.abs(v))/count,
                'median_absolute_TWD':None if not count else float(np.median(np.abs(v))),
                'p95_absolute_TWD':None if not count else float(np.quantile(np.abs(v),.95,method='linear')),
                'max_absolute_TWD':None if not count else float(np.max(np.abs(v)))}
    def summary(mask):
        count=int(mask.sum());out={'rows':count}
        for name,(e0,e1) in errors.items():
            a,b,change=total(e0[mask]),total(e1[mask]),total(deltas[name][mask])
            out[name]={'control_MAE_TWD':None if not count else a/count,
                       'candidate_MAE_TWD':None if not count else b/count,
                       'paired_delta_TWD':None if not count else change/count,
                       'control_abs_error_sum_TWD':a,'candidate_abs_error_sum_TWD':b,
                       'error_delta_sum_TWD':change,'contribution_to_overall_delta_TWD':change/n}
        ts=total(transfer[mask])
        out['transfer_effect_TWD']=None if not count else out['final']['paired_delta_TWD']-out['spatial']['paired_delta_TWD']
        out['transfer_error_sum_TWD']=ts;out['transfer_contribution_to_overall_TWD']=ts/n
        if count and not near(out['transfer_effect_TWD'],ts/count,sum(total(np.abs(v[mask])) for v in deltas.values())/count):
            raise ValueError('Transfer mean/sum reconciliation failed')
        out['predshift']={'spatial':shift_summary(shift_s,mask),'final':shift_summary(shift_f,mask)}
        return out
    unequal=lo<hi
    def regions(s):return {'below':s<lo,'within':(s>=lo)&(s<=hi),'above':s>hi}
    c0,c1=regions(s0),regions(s1)
    masks={f'{a}->{b}':unequal&c0[a]&c1[b] for a in REGIONS for b in REGIONS}
    masks['fixed_equal_bounds']=lo==hi
    if not np.all(sum(m.astype(np.int8) for m in masks.values())==1):raise ValueError('Transition partition failed')
    signs={'improved':deltas['spatial']<0,'unchanged':deltas['spatial']==0,'worsened':deltas['spatial']>0}
    overall=summary(np.ones(n,dtype=bool))
    cells={name:summary(mask) for name,mask in masks.items()}
    groups={name:summary(mask) for name,mask in signs.items()}
    for partition in (cells,groups):
        if sum(item['rows'] for item in partition.values())!=n:raise ValueError('Partition rows do not reconstruct overall')
        for layer,(e0,e1) in errors.items():
            for key,values in (('control_abs_error_sum_TWD',e0),('candidate_abs_error_sum_TWD',e1),('error_delta_sum_TWD',deltas[layer])):
                if not near(sum(item[layer][key] for item in partition.values()),overall[layer][key],total(np.abs(values))):
                    raise ValueError('Partition error sums do not reconstruct overall')
            if not near(sum(item[layer]['contribution_to_overall_delta_TWD'] for item in partition.values()),
                        overall[layer]['paired_delta_TWD'],total(np.abs(deltas[layer]))/n):
                raise ValueError('Partition contributions do not reconstruct overall MAE delta')
        if not near(sum(item['transfer_error_sum_TWD'] for item in partition.values()),overall['transfer_error_sum_TWD'],total(np.abs(transfer))):
            raise ValueError('Partition transfer sums do not reconstruct overall')
        if not near(sum(item['transfer_contribution_to_overall_TWD'] for item in partition.values()),overall['transfer_effect_TWD'],sum(total(np.abs(v)) for v in deltas.values())/n):
            raise ValueError('Partition transfer contributions do not reconstruct overall')
    return {'rows':n,'overall':overall,'transition_cells':cells,'error_sign_groups':groups,
        'rules':{'transition_order':list(TRANSITIONS),'within_includes_both_boundaries':True,
                 'equal_bounds_separate':True,'empty_cells_retained':True,'paired_delta':'candidate error minus control error; negative is improvement',
                 'transfer_effect':'final paired MAE delta minus spatial paired MAE delta',
                 'contribution_denominator':'all input rows','sign_group_threshold':0,
                 'contraction_tolerance':'8 * float64 eps * max(1,abs(spatials),abs(bounds),abs(finals)) per row'},
        'checks':{'final_projection_bitwise_exact':True,'prediction_shift_nonexpansive_with_tolerance':True,
                  'transition_rows_and_sums_reconstruct_overall':True,'sign_rows_and_sums_reconstruct_overall':True,
                  'maximum_observed_contraction_excess_TWD':float(np.max(excess)),
                  'maximum_contraction_tolerance_TWD':float(np.max(tolerance))},
        'limitations':['This examines already-saved projections; it is not a new clipping/cap rule or acceptance gate.',
                       'Prediction-shift contraction does not imply the same bound on paired error gains or preservation of improvement direction.',
                       'Transition/sign groups are descriptive, not causal effects; no threshold, seed or local-rule search.']}


def self_check():
    y=np.array([12.,-2.,5.,4.,5.,2.,4.,1.]);s0=np.array([20.,-10.,-5.,5.,0.,-10.,13.,-5.])
    s1=np.array([9.,1.,15.,-3.,10.,10.,-3.,-2.]);lo=np.array([0.,0.,0.,0.,0.,3.,0.,0.])
    hi=np.array([10.,10.,10.,10.,10.,3.,10.,10.])
    f0=np.median(np.column_stack([s0,lo,hi]),axis=1);f1=np.median(np.column_stack([s1,lo,hi]),axis=1)
    args=(y,s0,s1,lo,hi,f0,f1);before=[a.tobytes() for a in args]
    r=projection_review(*args)
    assert all(a.tobytes()==b for a,b in zip(args,before))
    assert list(r['transition_cells'])==list(TRANSITIONS) and len(r['transition_cells'])==10
    assert r['transition_cells']['fixed_equal_bounds']['rows']==1
    assert r['transition_cells']['within->within']['rows']==1 # both boundary ties belong within
    assert r['transition_cells']['below->above']['rows']==r['transition_cells']['above->below']['rows']==1
    empty=r['transition_cells']['above->above'];assert empty['rows']==0
    assert empty['spatial']['paired_delta_TWD'] is None and empty['final']['contribution_to_overall_delta_TWD']==0
    flip=r['transition_cells']['above->within']
    assert flip['rows']==1 and flip['spatial']['paired_delta_TWD']==-5. and flip['final']['paired_delta_TWD']==1.
    assert flip['transfer_effect_TWD']==6.
    assert all(sum(cell['rows'] for cell in r[k].values())==len(y) for k in ('transition_cells','error_sign_groups'))
    for k in ('spatial','final'):
        assert np.isclose(sum(v[k]['contribution_to_overall_delta_TWD'] for v in r['transition_cells'].values()),r['overall'][k]['paired_delta_TWD'])
    assert json.loads(json.dumps(r,allow_nan=False))==r
    invalid=[]
    a=list(args);a[0]=y.astype(np.float32);invalid.append(a)
    a=list(args);a[0]=y.copy();a[0][0]=np.nan;invalid.append(a)
    a=list(args);a[1]=s0[:-1];invalid.append(a)
    a=list(args);a[3]=hi+1;invalid.append(a)
    a=list(args);a[6]=f1.copy();a[6][0]+=1;invalid.append(a)
    for a in invalid:
        try:projection_review(*a)
        except ValueError:pass
        else:raise AssertionError('Invalid input accepted')
    return {'status':'passed','rows':len(y),'transition_cells':10,
            'checks':['median_equals_projection','boundary_ties_within','both_direction_bound_crossing','equal_bounds_separate',
                      'empty_cells_retained','rows_and_error_contribution_reconciliation','inputs_unchanged','JSON_roundtrip',
                      'float32_nonfinite_shape_bounds_final_mismatch_rejected','spatial_gain_can_reverse_after_projection'],
            'reversal_example':{'y':12.,'control_spatial':20.,'candidate_spatial':9.,'lower':0.,'upper':10.,
                                'control_final':10.,'candidate_final':9.,'spatial_error_delta':-5.,'final_error_delta':1.,'transfer_effect':6.},
            'model_fits':0,'model_predict':0,'GPU_calls':0,'real_arrays_read':False,'numpy':np.__version__}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--self-check',action='store_true',required=True)
    parser.add_argument('--output',type=Path,required=True);args=parser.parse_args()
    receipt=args.output.with_suffix('.sha256')
    if args.output.exists() or receipt.exists():raise FileExistsError('Preserve existing selfcheck evidence')
    tick=time.perf_counter();result=self_check();result['elapsed_seconds']=time.perf_counter()-tick
    result['source_sha256']=hashlib.sha256(Path(__file__).read_bytes()).hexdigest();result['exit_code']=0
    text=json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False)+'\n'
    with args.output.open('x',encoding='utf-8',newline='\n') as f:f.write(text)
    digest=hashlib.sha256(args.output.read_bytes()).hexdigest()
    with receipt.open('x',encoding='ascii',newline='\n') as f:f.write(digest+'\n')
    print(json.dumps({'status':'passed','source_sha256':result['source_sha256'],'result_sha256':digest,
                      'elapsed_seconds':result['elapsed_seconds'],'output':str(args.output)},ensure_ascii=False),flush=True)
