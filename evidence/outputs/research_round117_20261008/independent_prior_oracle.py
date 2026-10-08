"""Independent artificial-data arithmetic for R117; no real X/NPZ/prices/GPU.

The oracle uses Python tuples/math and explicit four-cell routing. It never
calls the original key builder, date-weight helper or weighted-median helper.
Actual candidate calls receive artificial labels only.
"""
import hashlib
import json
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import numpy as np
import pandas as pd
import round117_transaction_prior as candidate

LEVELS=(('address_city','address_district','transaction_type','building_type'),
        ('address_city','transaction_type','building_type'),
        ('transaction_type','building_type'),('transaction_type',))


def need(value,message):
    if not value:raise AssertionError(message)


def weights(rows):
    valid=[math.isfinite(r['trade_day_index']) and r['trade_date_invalid']==0
           and r['trade_date_out_of_range']==0 for r in rows]
    anchor=max(r['trade_day_index'] for r,v in zip(rows,valid) if v)
    unnormalized=[2.**((r['trade_day_index']-anchor)/365.) for r,v in zip(rows,valid) if v]
    mean=sum(unnormalized)/len(unnormalized)
    result=[2.**((r['trade_day_index']-anchor)/365.)/mean if v else 1. for r,v in zip(rows,valid)]
    return result,anchor,mean


def median(values,ws):
    ordered=sorted(enumerate(zip(values,ws)),key=lambda x:(x[1][0],x[0]))
    maximum=max(ws);scaled=[(float(v),float(w)/maximum) for _,(v,w) in ordered]
    total=sum(w for _,w in scaled);half=total/2.;running=0.
    for i,(v,w) in enumerate(scaled):
        running+=w
        if running>=half:
            return v+(scaled[i+1][0]-v)/2. if running==half and i+1<len(scaled) else v
    raise AssertionError('No weighted median')


def oracle(rows,units,held):
    selected=[i for i in range(len(rows)) if not held[i]]
    references=[rows[i] for i in selected]
    ws,anchor,mean=weights(references)
    prior=[];levels=[]
    for i in np.flatnonzero(held):
        for level,columns in enumerate(LEVELS):
            positions=[j for j,k in enumerate(selected) if tuple(rows[k][c] for c in columns)==tuple(rows[i][c] for c in columns)]
            if len(positions)>=30:break
        else:
            positions=list(range(len(selected)));level=4
        prior.append(median([units[selected[j]] for j in positions],[ws[j] for j in positions]));levels.append(level)
    return np.asarray(prior,np.float64),levels,anchor,mean


def fixture():
    rows=[];unit=[]
    for city,district,kind,building,n,value in [('A','D0','house','apartment',90,10.),
        ('A','D1','house','apartment',29,100.),('B','D0','house','apartment',29,200.),
        ('C','D0','house','villa',29,300.),('D','D0','land','raw',29,400.)]:
        for j in range(n):
            i=len(rows)
            rows.append({'address_city':city,'address_district':district,'transaction_type':kind,'building_type':building,
                'project_name':'shared_same_project','address_raw':'__MISSING__','address_road':'__MISSING__','address_section':'__MISSING__',
                'building_area':100.,'parking_area':0.,'trade_day_index':1000.+365*(i%5),
                'trade_date_invalid':float(i%7==0),'trade_date_out_of_range':0.})
            unit.append(value+j)
    rows[1]['trade_day_index']=10000.
    return pd.DataFrame(rows),np.asarray(unit,np.float64)


def run():
    checks=[]
    def mark(name,condition):need(bool(condition),name);checks.append(name)
    mark('manual_exact_half_midpoint',median([1.,9.],[1.,1.])==5.)
    mark('manual_stable_repeated_values',median([1.,2.,2.,9.],[1.,1.,1.,1.])==2.)
    w,anchor,mean=weights([{'trade_day_index':1000.,'trade_date_invalid':0.,'trade_date_out_of_range':0.},
        {'trade_day_index':1365.,'trade_date_invalid':0.,'trade_date_out_of_range':0.},
        {'trade_day_index':math.nan,'trade_date_invalid':1.,'trade_date_out_of_range':0.}])
    mark('hand_neutral_date_and_mean_valid_only',w==[2/3,4/3,1.] and anchor==1365. and mean==.75)
    X,units=fixture();rows=X.to_dict('records');n=len(X);indices=np.arange(1000,1000+n,dtype=np.int64)
    expected_folds=np.empty(n,np.int64);expected_folds[np.random.default_rng(42).permutation(n)]=np.arange(n)%5
    actual,full,folds,evidence=candidate.fit_oof_prior(X,units*100,indices)
    mark('independent_balanced_permutation_rule',np.array_equal(folds,expected_folds))
    used=set();anchors=[]
    for fold in range(5):
        held=expected_folds==fold
        expected,levels,anchor,mean=oracle(rows,units,held)
        mark('manual_four_cell_OOF_values_fold_'+str(fold),np.array_equal(actual[held],expected))
        mark('manual_complement_anchor_fold_'+str(fold),evidence['folds'][fold]['date_support']['max_valid_reference_trade_day_index']==anchor)
        mark('manual_complement_normalization_fold_'+str(fold),math.isclose(evidence['folds'][fold]['date_support']['normalization_mean_valid_reference_only'],mean,rel_tol=1e-14,abs_tol=0.))
        used.update(levels);anchors.append(anchor)
    mark('all_four_levels_and_global_exercised',used==set(range(5)))
    mark('complement_specific_anchor_changes_when_latest_held',len(set(anchors))==2)
    held=folds==int(folds[1]);changedX=X.copy(deep=True);changedY=units*100
    changedX.loc[held,'trade_day_index']+=5000.;changedY[held]*=31.
    revised,_,samefolds,revised_evidence=candidate.fit_oof_prior(changedX,changedY,indices)
    mark('held_entire_fold_label_date_invariance',np.array_equal(actual[held],revised[held]) and np.array_equal(folds,samefolds)
         and evidence['folds'][int(folds[1])]==revised_evidence['folds'][int(folds[1])])
    same_project=X['address_city'].eq('A').to_numpy();held=folds==0
    exposed=same_project & ~held
    changedY=units*100;changedY[exposed]*=100.
    changed,_,_,_=candidate.fit_oof_prior(X,changedY,indices)
    mark('other_fold_same_project_references_allowed_and_can_affect_prior',np.any(actual[same_project&held]!=changed[same_project&held]))
    # Exact raw30 versus raw29 and exact-half arithmetic under equal dates.
    tiny=X.iloc[:31].copy();tiny.loc[:,'trade_day_index']=0.;tiny.loc[:,'trade_date_invalid']=0.
    tiny.loc[30,'address_district']='rare';tiny_units=np.arange(1.,32.)
    _,tiny_full,_,_=candidate.fit_oof_prior(tiny,tiny_units*100,np.arange(31,dtype=np.int64))
    query=tiny.iloc[[0,30]].drop(columns=list(candidate.original.DATE_COLUMNS))
    predicted=candidate.predict_prior(tiny_full,query)
    mark('raw30_finest_cell_exact_half_15_5',predicted[0]==15.5)
    mark('raw1_finest_falls_to_city31_median16',predicted[1]==16.)
    short=tiny.iloc[:29].copy()
    _,short_full,_,_=candidate.fit_oof_prior(short,np.arange(1.,30.)*100,np.arange(29,dtype=np.int64))
    mark('raw29_all_cells_absent_global_median15',all(x['cell_count']==0 for x in short_full['levels'])
         and np.array_equal(candidate.predict_prior(short_full,short),np.full(29,15.)))
    mark('legacy_fullfit_state_not_relabelled_as_transaction_policy',full['rules']==candidate.original.RULES
         and evidence['fold_policy']=='transaction-permutation-5-fold')
    return {'status':'passed','checks':checks,'check_count':len(checks),'oracle_uses_original_key_weight_or_median_helpers':False,
        'source_sha256':{**candidate.source_hashes(),Path(__file__).relative_to(ROOT).as_posix():hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
        'artificial_rows':n,'real_prior_builds':0,'real_arrays_or_query_y_read':False,'new_model_fits':0,'GPU_work':False}


if __name__=='__main__':print(json.dumps(run(),ensure_ascii=False,sort_keys=True))
