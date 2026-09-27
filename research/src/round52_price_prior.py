"""R52 supervised unit-price priors: five group-held-out fit folds, full-fit query state."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
import numpy as np
import pandas as pd
import v3_cluster_stability as grouping
from round19_unit_geo import effective_area

LEVELS=[['address_city','address_district','transaction_type','building_type'],
        ['address_city','transaction_type','building_type'],['transaction_type','building_type'],['transaction_type']]
RULES={'folds':5,'seed':42,'minimum_cell_rows':30,'unit':'TWD per square metre of effective area',
       'area':'finite positive building_area else finite positive parking_area else 1',
       'target':'y_TWD / effective_area','levels':LEVELS,'cell_estimator':'median; fallback if fewer than 30 rows',
       'global':'median of all available reference rows; always positive finite',
       'fold_hash':"int.from_bytes(sha256(('42|'+existing_cluster_key).encode('utf-8')).digest()[:8],'little') % 5",
       'training':'exclude complete held fold, including every row of its project/fallback groups',
       'query':'full fit rows only','time_multiplier':False,'clipping':False}
ROOT=Path(__file__).resolve().parents[1]


def need(ok,message):
    if not ok:raise ValueError(message)


def values_sha(values,dtype='<f8'):
    return hashlib.sha256(np.asarray(values,dtype=dtype).tobytes(order='C')).hexdigest()


def _keys(X):
    need(isinstance(X,pd.DataFrame) and X.columns.is_unique and X.index.is_unique,'Unique X columns/index required')
    columns=list(dict.fromkeys(c for level in LEVELS for c in level))
    need(set(columns)<=set(X.columns),'Missing prior category fields')
    normalized={c:grouping.norm(X[c]).to_numpy() for c in columns}
    return [np.asarray([json.dumps(row,ensure_ascii=False,separators=(',',':'))
                        for row in zip(*(normalized[c] for c in level))],dtype=object) for level in LEVELS]


def _build(keys,units,mask):
    values=units[mask]
    need(len(values)>0 and np.isfinite(values).all() and (values>0).all(),'Reference unit prices must be positive finite and nonempty')
    global_median=float(np.median(values));need(np.isfinite(global_median) and global_median>0,'Invalid global prior')
    levels=[]
    for columns,key in zip(LEVELS,keys):
        summary=pd.DataFrame({'key':key[mask],'unit':values}).groupby('key',sort=True)['unit'].agg(['size','median'])
        selected=summary.loc[summary['size']>=30]
        need(np.isfinite(selected['median']).all() and (selected['median']>0).all(),'Invalid cell prior')
        levels.append({'columns':list(columns),'cell_count':len(selected),
                       'minimum_observed_support':int(selected['size'].min()) if len(selected) else None,'cells':{str(k):{'rows':int(row['size']),'median':float(row['median'])}
                       for k,row in selected.iterrows()}})
    return {'rules':copy.deepcopy(RULES),'fit_rows':int(mask.sum()),'global_median':global_median,'levels':levels,
            'uses_fit_labels':True,'uses_query_labels':False,'state_role':'full_fit_for_query'}


def _predict(state,keys,return_diagnostics=False):
    need(state['rules']==RULES and len(state['levels'])==len(LEVELS),'Prior rule/schema changed')
    global_median=float(state['global_median'])
    need(np.isfinite(global_median) and global_median>0,'Invalid saved global prior')
    result=np.full(len(keys[0]),global_median,dtype=np.float64);unassigned=np.ones(len(result),dtype=bool)
    counts={'|'.join(columns):0 for columns in LEVELS};counts['global']=0
    for columns,key,level in zip(LEVELS,keys,state['levels']):
        need(level['columns']==columns,'Prior hierarchy changed')
        for i in np.flatnonzero(unassigned):
            cell=level['cells'].get(key[i])
            if cell is not None:
                need(cell['rows']>=30 and np.isfinite(cell['median']) and cell['median']>0,'Invalid saved cell prior')
                result[i]=cell['median'];unassigned[i]=False;counts['|'.join(columns)]+=1
    need(np.isfinite(result).all() and (result>0).all(),'Invalid predicted prior')
    counts['global']=int(unassigned.sum())
    return (result,counts) if return_diagnostics else result


def predict_prior(state,Xquery,return_diagnostics=False):
    """Ordered float64 TWD/m2 priors; no query labels or vocabulary fitting."""
    return _predict(state,_keys(Xquery),return_diagnostics=return_diagnostics)


def fit_oof_prior(Xfit,yfit,source_indices):
    """Return (OOF TWD/m2 float64, JSON-safe full-fit state, int64 folds, evidence).

    source_indices must be canonical fit indices in exactly Xfit/yfit row order.
    This is grouped OOF, not chronological OOF; all reference rows are from fit.
    """
    keys=_keys(Xfit);n=len(Xfit);indices=np.asarray(source_indices)
    need(n>0 and indices.dtype.kind in 'iu' and indices.shape==(n,) and (indices>=0).all()
         and len(np.unique(indices))==n,'Invalid ordered canonical source indices')
    if isinstance(yfit,pd.Series):need(yfit.index.equals(Xfit.index),'Fit target Series index differs')
    y=np.asarray(yfit,dtype=np.float64)
    need(y.shape==(n,) and np.isfinite(y).all() and (y>0).all(),'Fit prices must be positive finite and aligned')
    area=effective_area(Xfit)
    with np.errstate(over='raise',invalid='raise',divide='raise'):units=y/area
    need(np.isfinite(units).all() and (units>0).all(),'Invalid fit unit prices')
    groups,fallback=grouping.cluster_keys(Xfit,indices)
    assignments={key:int.from_bytes(hashlib.sha256(('42|'+key).encode('utf-8')).digest()[:8],'little')%5
                 for key in np.unique(groups)}
    folds=np.asarray([assignments[key] for key in groups],dtype=np.int64)
    oof=np.empty(n,dtype=np.float64);fold_evidence=[]
    for fold in range(5):
        held=folds==fold;reference=~held
        record={'fold':fold,'held_rows':int(held.sum()),'reference_rows':int(reference.sum()),
                'held_groups':int(len(np.unique(groups[held]))),'reference_groups':int(len(np.unique(groups[reference]))),
                'fit_rows':int(reference.sum()),'query_rows':int(held.sum())}
        if held.any():
            state=_build(keys,units,reference)
            oof[held],record['held_prior_level_counts']=_predict(state,[key[held] for key in keys],return_diagnostics=True)
            record['reference_global_median']=state['global_median']
        else:
            record['reference_global_median']=None
            record['held_prior_level_counts']={**{'|'.join(c):0 for c in LEVELS},'global':0}
        fold_evidence.append(record)
    state=_build(keys,units,np.ones(n,dtype=bool))
    source_paths=[Path(__file__),Path(grouping.__file__),ROOT/'src/round19_unit_geo.py']
    evidence={'rules':copy.deepcopy(RULES),'fit_rows':n,'groups':len(assignments),'fallback_counts':fallback,
              'full_state_level_support':[{'columns':level['columns'],'cell_count':level['cell_count'],
                  'minimum_observed_support':level['minimum_observed_support']} for level in state['levels']],
              'folds':fold_evidence,'training_prior_role':'out_of_fold; entire held group excluded',
              'query_prior_role':'full fit state','uses_fit_labels':True,'uses_query_labels':False,'chronological_oof':False,
              'fit_target_sha256':values_sha(y),'effective_area_sha256':values_sha(area),'fit_unit_sha256':values_sha(units),
              'source_indices_sha256':values_sha(indices,'<i8'),'fold_ids_sha256':values_sha(folds,'<i8'),
              'oof_prior_sha256':values_sha(oof),'group_keys_sha256':hashlib.sha256(json.dumps(groups.tolist(),ensure_ascii=False,separators=(',',':')).encode()).hexdigest(),
              'state_content_sha256':hashlib.sha256(json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
              'source_sha256':{p.resolve().relative_to(ROOT).as_posix():hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}}
    return oof,state,folds,evidence


def self_check():
    n=360
    X=pd.DataFrame({'project_name':[f'project_{i//6}' for i in range(n)],'address_city':['city']*n,
       'address_district':['district']*n,'address_raw':['__MISSING__']*n,'address_road':['__MISSING__']*n,
       'address_section':['__MISSING__']*n,'transaction_type':['house']*n,'building_type':['apartment']*n,
       'building_area':np.full(n,100,dtype=np.float32),'parking_area':np.zeros(n,dtype=np.float32)})
    y=pd.Series((np.arange(n,dtype=np.float64)+100)*100,index=X.index);before=X.copy(deep=True);original_y=y.copy()
    indices=np.arange(1000,1000+n,dtype=np.int64)
    oof,state,folds,evidence=fit_oof_prior(X,y,indices);need(set(folds)==set(range(5)),'Synthetic fixture must cover five folds')
    changed=y.copy();changed.loc[folds==0]*=7
    revised,_,samefolds,_=fit_oof_prior(X,changed,indices)
    need(np.array_equal(folds,samefolds) and oof[folds==0].tobytes()==revised[folds==0].tobytes(),'Held-fold labels leaked into their prior')
    for start in range(0,n,6):need(len(set(folds[start:start+6]))==1,'A project crossed folds')
    query=X.iloc[:3].copy();query.loc[:,['address_city','address_district','transaction_type','building_type']]='unknown'
    result=predict_prior(state,query);need(np.array_equal(result,np.full(3,state['global_median'])),'Unknown keys did not reach global fallback')
    restored=json.loads(json.dumps(state,allow_nan=False))
    need(predict_prior(state,X).tobytes()==predict_prior(restored,X).tobytes(),'JSON state roundtrip differs')
    need(X.equals(before) and y.equals(original_y),'Fit inputs mutated')
    return {'status':'passed','held_fold_label_invariance':True,'same_project_single_fold':True,
            'unknown_global_fallback':True,'json_roundtrip_exact':True,'fit_inputs_unchanged':True,
            'synthetic_rows':n,'real_data_read':False,'model_fits':0}


if __name__=='__main__':print(json.dumps(self_check(),ensure_ascii=False))
