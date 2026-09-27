"""Compare fixed candidate predictions and cross-fit coarse blend weights by project."""
from __future__ import annotations
import argparse
import json
from itertools import combinations
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from model_common import ROOT, load_data, temporal_indices, mae

OUT = ROOT / 'outputs' / 'round2' / 'blending'
BASE = ['catboost_raw','catboost_unit','catboost_structural','lightgbm_l1_raw63','lightgbm_l1_unit63']
MISSING = '__MISSING__'


def read_array(path, rows):
    p = np.asarray(np.load(ROOT/path, allow_pickle=False), dtype='float64')
    if p.shape != (rows,) or not np.isfinite(p).all():
        raise ValueError(f'Invalid prediction array: {path}')
    return np.maximum(p, 0)


def run(include_new=True):
    OUT.mkdir(parents=True,exist_ok=True)
    d=load_data()
    fit, ti=temporal_indices(d,'2024-04-01',end='2024-08-10')
    xt=d['X_train'].iloc[ti]
    y=d['y_train'][ti]
    seen=xt.project_name.isin(set(d['X_train'].iloc[fit].project_name)-{MISSING}).to_numpy()
    vs=d['X_valid'].project_name.isin(set(d['X_train'].project_name)-{MISSING}).to_numpy()
    ts=d['X_test'].project_name.isin((set(d['X_train'].project_name)|set(d['X_valid'].project_name))-{MISSING}).to_numpy()
    rate=float((~ts).mean())
    catalog=[]
    for name in BASE:
        catalog.append({'name':name,'valid_predictions':f'outputs/experiments/{name}_valid.npy',
                        'temporal_predictions':f'outputs/experiments/{name}_temporal_valid.npy',
                        'temporal_indices':f'outputs/experiments/{name}_temporal_indices.npy',
                        'metadata_path':f'outputs/experiments/{name}.json'})
    if include_new:
        for branch in ['catboost','lightgbm','neighborhood']:
            path=ROOT/'outputs'/'round2'/branch/'candidates.json'
            if path.is_file():
                items=json.loads(path.read_text(encoding='utf-8-sig'))
                if not isinstance(items,list): raise ValueError(f'Expected a list: {path}')
                catalog.extend(items)
    if len({c['name'] for c in catalog})!=len(catalog): raise ValueError('Duplicate candidate names')
    temporal={}
    validation={}
    for c in catalog:
        ix=np.load(ROOT/c['temporal_indices'],allow_pickle=False)
        if not np.array_equal(ix,ti): raise ValueError(f"Misaligned temporal indices: {c['name']}")
        temporal[c['name']]=read_array(c['temporal_predictions'],len(y))
        if c.get('valid_predictions'):
            validation[c['name']]=read_array(c['valid_predictions'],len(d['y_valid']))
    names=list(temporal)
    parts=[[(name,1.)] for name in names]
    parts.extend([[(a,w),(b,1-w)] for a,b in combinations(names,2) for w in [.25,.5,.75]])
    def combine(scheme, source):
        if any(n not in source for n,_ in scheme): return None
        return sum(w*source[n] for n,w in scheme)
    predictions=np.stack([combine(p,temporal) for p in parts])
    errors=np.abs(predictions-y)
    def reweight(p,mask=None):
        mask=np.ones(len(y),dtype=bool) if mask is None else mask
        return float((1-rate)*np.mean(np.abs(p[mask&seen]-y[mask&seen]))+rate*np.mean(np.abs(p[mask&~seen]-y[mask&~seen])))
    def best(mask,gate=None):
        if gate is None:
            loss=(1-rate)*errors[:,mask&seen].mean(axis=1)+rate*errors[:,mask&~seen].mean(axis=1)
        else:
            loss=errors[:,mask&(seen==gate)].mean(axis=1)
        return int(np.argmin(loss))
    full=np.ones(len(y),dtype=bool)
    global_index=best(full)
    gate_indices={True:best(full,True),False:best(full,False)}
    gp=predictions[global_index].copy()
    gated=np.where(seen,predictions[gate_indices[True]],predictions[gate_indices[False]])
    group=xt['project_name'].astype(str).copy()
    fallback=next((c for c in ['address_road','road_name','address_raw'] if c in xt),None)
    if fallback: group=group.where(group!=MISSING,xt[fallback].astype(str))
    group=xt['address_city'].astype(str)+'|'+xt['address_district'].astype(str)+'|'+group
    cv_global=np.zeros(len(y)); cv_gated=np.zeros(len(y)); folds=np.full(len(y),-1,dtype='int8')
    fold_records=[]
    for f,(learn,check) in enumerate(GroupKFold(n_splits=5).split(np.zeros(len(y)),groups=group.to_numpy())):
        mask=np.zeros(len(y),dtype=bool); mask[learn]=True
        gi=best(mask); si=best(mask,True); ui=best(mask,False)
        cv_global[check]=predictions[gi,check]
        cv_gated[check]=np.where(seen[check],predictions[si,check],predictions[ui,check])
        folds[check]=f
        fold_records.append({'fold':f,'rows':len(check),'global':parts[gi], 'seen':parts[si],'unseen':parts[ui]})
    baseline=.75*temporal['catboost_raw']+.25*temporal['lightgbm_l1_unit63']
    baseline_valid=.75*validation['catboost_raw']+.25*validation['lightgbm_l1_unit63']
    candidate_rows=[]
    for c in catalog:
        name=c['name']; p=temporal[name]
        candidate_rows.append({'name':name,'temporal_mae':mae(y,p),'seen_mae':mae(y[seen],p[seen]),
                               'unseen_mae':mae(y[~seen],p[~seen]),'reweighted_temporal_mae':reweight(p),
                               'validation_mae':mae(d['y_valid'],validation[name]) if name in validation else None})
    configs={
        'global':{'seen':parts[global_index],'unseen':parts[global_index]},
        'project_gated':{'seen':parts[gate_indices[True]],'unseen':parts[gate_indices[False]]},
    }
    strategy=[]
    for label,p,cv in [('global',gp,cv_global),('project_gated',gated,cv_gated)]:
        config=configs[label]
        a=combine(config['seen'],validation); b=combine(config['unseen'],validation)
        pv=np.where(vs,a,b) if a is not None and b is not None else None
        row={'strategy':label,'gates':config,'temporal_mae':mae(y,p),'reweighted_temporal_mae':reweight(p),
             'crossfit_temporal_mae':mae(y,cv),'crossfit_reweighted_temporal_mae':reweight(cv),
             'seen_mae':mae(y[seen],p[seen]),'unseen_mae':mae(y[~seen],p[~seen]),
             'validation_mae':mae(d['y_valid'],pv) if pv is not None else None,
             'clean_validation_mae':mae(d['y_valid'][d['valid_clean_mask']],pv[d['valid_clean_mask']]) if pv is not None else None,
             'needs_original_validation':[n for gate in config.values() for n,_ in gate if n not in validation]}
        strategy.append(row)
        np.save(OUT/f'{label}_temporal.npy',p)
        np.save(OUT/f'{label}_crossfit_temporal.npy',cv)
        if pv is not None: np.save(OUT/f'{label}_valid.npy',pv)
    strategy.sort(key=lambda r:r['crossfit_reweighted_temporal_mae'])
    chosen=strategy[0]
    gap=(reweight(baseline)-chosen['crossfit_reweighted_temporal_mae'])/reweight(baseline)
    out={'candidate_count':len(names),'coarse_schemes':len(parts),'test_unseen_project_rate':rate,
         'baseline_v1':{'validation_mae':mae(d['y_valid'],baseline_valid),'temporal_mae':mae(y,baseline),'reweighted_temporal_mae':reweight(baseline)},
         'candidates':candidate_rows,'strategies':strategy,'chosen_strategy':chosen['strategy'],
         'relative_crossfit_improvement_over_v1':gap,'fold_choices':fold_records,
         'crossfit_caveat':'Blend weights cross-fit by project. Base models already used this temporal holdout for early stopping; this is a stability diagnostic, not an untouched test score.',
         'catalog':catalog}
    out['crossfit_group_definition']='address_city | address_district | project_name; missing project uses road/address'
    reference_path=OUT/'baseline_crossfit_reference.json'
    if not include_new:
        reference_path.write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    if reference_path.is_file():
        reference=json.loads(reference_path.read_text(encoding='utf-8'))
        reference_score=next(r['crossfit_reweighted_temporal_mae'] for r in reference['strategies'] if r['strategy']=='global')
        out['baseline_global_strategy_crossfit_mae']=reference_score
        out['relative_crossfit_strategy_improvement']=(reference_score-chosen['crossfit_reweighted_temporal_mae'])/reference_score
    (OUT/'comparison.json').write_text(json.dumps(out,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
    pd.DataFrame(candidate_rows).to_csv(OUT/'candidate_metrics.csv',index=False,encoding='utf-8-sig')
    np.save(OUT/'temporal_indices.npy',ti); np.save(OUT/'project_folds.npy',folds)
    print(json.dumps({'candidates':len(names),'chosen':chosen,'relative_crossfit_improvement':gap},ensure_ascii=True),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--baseline-only',action='store_true')
    args=parser.parse_args()
    run(not args.baseline_only)