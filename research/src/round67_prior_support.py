"""R67 raw support of the selected saved R60 prior cell; fit X/folds only."""
from __future__ import annotations
import copy
import json
import time
import numpy as np
import pandas as pd
import round60_recency_price_prior as prior

LEVELS=prior.LEVELS
MIN_ROWS=30
LEVEL_NAMES=['|'.join(c) for c in LEVELS]+['global']
sha_values=prior.values_sha


def _tables(keys,mask):
    return [{str(k):int(n) for k,n in pd.Series(key[mask]).value_counts(sort=False).items()
             if n>=MIN_ROWS} for key in keys]


def _level_summary(tables):
    return [{'columns':list(columns),'cell_count':len(table),
             'minimum_observed_support':min(table.values(),default=None)}
            for columns,table in zip(LEVELS,tables)]


def _lookup(keys,tables,global_rows):
    assert len(keys)==len(tables)==len(LEVELS) and global_rows>0
    n=len(keys[0]);counts=np.full(n,global_rows,dtype=np.int64)
    assigned=np.zeros(n,dtype=bool);usage={name:0 for name in LEVEL_NAMES}
    for name,key,table in zip(LEVEL_NAMES,keys,tables):
        for i in np.flatnonzero(~assigned):
            value=table.get(key[i])
            if value is not None:
                assert MIN_ROWS<=value<=global_rows
                counts[i]=value;assigned[i]=True;usage[name]+=1
    usage['global']=int((~assigned).sum())
    assert sum(usage.values())==n and (counts>0).all()
    return counts,usage


def _summary(values):
    return {'rows':len(values),'min':int(values.min()),'max':int(values.max()),
            'mean':float(values.mean()),'quantiles':{str(q):float(np.quantile(values,q))
                for q in (0,.25,.5,.75,.9,.99,1)},'values_sha256':sha_values(values,'<i8')}


def build_support(Xfit,Xquery,fit_indices,folds,state,evidence):
    """Return raw int64 train/query counts and evidence; no y or unit-price argument."""
    indices=np.asarray(fit_indices);folds=np.asarray(folds);n=len(Xfit)
    assert n>0 and len(Xquery)>0 and indices.shape==folds.shape==(n,)
    assert indices.dtype==folds.dtype==np.int64 and np.unique(indices).size==n
    assert set(np.unique(folds))<=set(range(5))
    assert sha_values(indices,'<i8')==evidence['source_indices_sha256']
    assert sha_values(folds,'<i8')==evidence['fold_ids_sha256']
    assert state['rules']==evidence['rules']==prior.RULES
    assert state['state_role']=='full_fit_for_query'
    assert state['global_raw_rows']==state['fit_rows']==evidence['fit_rows']==n
    assert not state['uses_query_labels'] and not evidence['uses_query_labels']
    keys=prior.original._keys(Xfit);query_keys=prior.original._keys(Xquery)
    saved_folds={record['fold']:record for record in evidence['folds']}
    assert set(saved_folds)==set(range(5))
    train=np.empty(n,dtype=np.int64);records=[]
    for fold in range(5):
        held=folds==fold;reference=~held;record=saved_folds[fold]
        assert record['held_rows']==record['query_rows']==int(held.sum())
        assert record['reference_rows']==record['fit_rows']==int(reference.sum())
        assert record['reference_indices_sha256']==sha_values(indices[reference],'<i8')
        if held.any():
            tables=_tables(keys,reference)
            assert _level_summary(tables)==record['reference_level_support']
            values,usage=_lookup([key[held] for key in keys],tables,int(reference.sum()))
            assert usage==record['held_prior_level_counts']
            train[held]=values
        else:
            usage={name:0 for name in LEVEL_NAMES}
            assert usage==record['held_prior_level_counts']
        records.append({'fold':fold,'held_rows':int(held.sum()),
                        'reference_rows':int(reference.sum()),'selected_level_rows':usage})
    full_tables=_tables(keys,np.ones(n,dtype=bool))
    assert _level_summary(full_tables)==evidence['full_state_level_support']
    assert len(state['levels'])==len(LEVELS)
    saved_tables=[]
    for columns,table,level in zip(LEVELS,full_tables,state['levels']):
        assert level['columns']==columns
        saved={key:int(cell['rows']) for key,cell in level['cells'].items()}
        assert saved==table
        assert level['cell_count']==len(saved)
        assert level['minimum_observed_support']==min(saved.values(),default=None)
        saved_tables.append(saved)
    query,usage=_lookup(query_keys,saved_tables,int(state['global_raw_rows']))
    assert train.shape==(n,) and query.shape==(len(Xquery),)
    assert (train>0).all() and (query>0).all()
    summary={'feature':'raw reference rows of the first matching R60 prior cell',
             'minimum_cell_rows':MIN_ROWS,'levels':copy.deepcopy(LEVELS),
             'global_fallback':'all reference rows of that fold/full fit',
             'train':_summary(train),'query':_summary(query),'folds':records,
             'query_selected_level_rows':usage,'saved_fold_level_counts_exact':True,
             'saved_full_state_all_cell_rows_exact':True,'saved_fold_reference_level_support_exact':True,
             'fit_indices_sha256':sha_values(indices,'<i8'),'fold_ids_sha256':sha_values(folds,'<i8'),
             'target_values_used':False,'query_population_counts_used':False,
             'prior_reestimated':False,'OOF_counts_rescaled':False,
             'counts_are_not_ESS_or_independent_groups_or_probability':True}
    return train,query,summary


def self_check():
    tick=time.monotonic();n=200;indices=np.arange(1000,1200,dtype=np.int64)
    folds=np.arange(n,dtype=np.int64)%5
    X=pd.DataFrame({'address_city':['A']*165+['B']*35,'address_district':['D']*n,
                    'transaction_type':['house']*n,'building_type':['apartment']*n})
    Q=pd.DataFrame({'address_city':['A','B','C','C'],'address_district':['D']*4,
                   'transaction_type':['house','house','house','unknown'],
                   'building_type':['apartment']*4})
    keys=prior.original._keys(X)
    full=[]
    for j,columns in enumerate(LEVELS):
        cells={keys[j][0]:{'rows':165},keys[j][165]:{'rows':35}} if j<2 else {keys[j][0]:{'rows':200}}
        full.append({'columns':list(columns),'cells':cells,'cell_count':len(cells),
                     'minimum_observed_support':min(v['rows'] for v in cells.values())})
    state={'rules':copy.deepcopy(prior.RULES),'state_role':'full_fit_for_query','global_raw_rows':200,
           'fit_rows':200,'uses_query_labels':False,'levels':full}
    usage={name:0 for name in LEVEL_NAMES};usage[LEVEL_NAMES[0]]=33;usage[LEVEL_NAMES[2]]=7
    fold_records=[]
    for fold in range(5):
        fold_records.append({'fold':fold,'held_rows':40,'query_rows':40,'reference_rows':160,'fit_rows':160,
          'reference_indices_sha256':sha_values(indices[folds!=fold],'<i8'),
          'reference_level_support':[{'columns':list(c),'cell_count':1,
            'minimum_observed_support':132 if j<2 else 160} for j,c in enumerate(LEVELS)],
          'held_prior_level_counts':dict(usage)})
    evidence={'rules':copy.deepcopy(prior.RULES),'fit_rows':n,'uses_query_labels':False,
              'source_indices_sha256':sha_values(indices,'<i8'),'fold_ids_sha256':sha_values(folds,'<i8'),
              'folds':fold_records,'full_state_level_support':[
                  {k:level[k] for k in ('columns','cell_count','minimum_observed_support')} for level in full]}
    train,query,summary=build_support(X,Q,indices,folds,state,evidence)
    assert np.array_equal(train,np.r_[np.full(165,132),np.full(35,160)])
    assert np.array_equal(query,[165,35,200,200])
    assert summary['query_selected_level_rows']['global']==1
    # Exact eligibility boundary independent of price values.
    boundary=_tables([np.array(['yes']*30+['no']*29,dtype=object)]*4,np.ones(59,dtype=bool))
    assert all(t=={'yes':30} for t in boundary)
    broken=copy.deepcopy(state);broken['levels'][0]['cells'][keys[0][0]]['rows']=166
    try:build_support(X,Q,indices,folds,broken,evidence)
    except AssertionError:pass
    else:raise AssertionError('Mismatched saved support accepted')
    return {'status':'passed','fold_exclusion_and_threshold_fallback':True,'full_query_and_global_fallback':True,
            'threshold_29_30_exact':True,'mismatched_state_rejected':True,'real_data_read':False,
            'model_fits':0,'model_predictions':0,'elapsed_seconds':time.monotonic()-tick}


if __name__=='__main__':print(json.dumps(self_check(),ensure_ascii=False))
