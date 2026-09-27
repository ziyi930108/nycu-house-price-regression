"""Fit-only numerical/category encoder for the unchanged R6 95-column input.

The caller supplies base.augment_features(X87, geo8) and base.categories(X87,cats21).
No targets, geography lookup, model, torch, device, or file loading occurs here.
"""
from __future__ import annotations
import hashlib
import json
from collections import Counter
import numpy as np
import pandas as pd

MIN_CATEGORY_COUNT=5
EPSILON=1e-6


def _need(ok,message):
    if not ok:raise ValueError(message)


def _content_sha(value):
    return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode('utf-8')).hexdigest()


def _frame(X,cats):
    _need(isinstance(X,pd.DataFrame) and len(X.columns)==95 and X.columns.is_unique,'Expected 95 distinct input columns')
    _need(all(isinstance(c,str) for c in X.columns),'Feature names must be strings')
    _need(not {'總價元','y','target','ID','id','編號'}&set(X.columns),'Targets/identifiers cannot enter encoder')
    _need(len(cats)==23 and len(set(cats))==23 and set(cats)<=set(X.columns),'Expected 23 categorical columns')
    numeric=[c for c in X.columns if c not in cats]
    _need(len(numeric)==72,'Expected 72 numeric columns')
    for c in numeric:
        _need(pd.api.types.is_numeric_dtype(X[c].dtype) and not pd.api.types.is_complex_dtype(X[c].dtype),
              'Numeric feature must contain real values: '+c)
    for c in cats:
        _need(X[c].map(lambda v:isinstance(v,str)).all(),'Category must use existing string representation: '+c)
    return numeric


def fit_encoder(X95,cats):
    """Fit pure-X state; numeric statistics use the median-imputed fit columns.

    Median uses finite fit values; a completely nonfinite fit column uses zero.
    IQR/std are computed after that imputation (population std, ddof=0).
    """
    cats=list(cats)
    numeric=_frame(X95,cats)
    _need(len(X95)>0,'Cannot fit an empty encoder')
    stats=[]
    for c in numeric:
        values=X95[c].to_numpy(dtype=np.float64,copy=True)
        finite=np.isfinite(values)
        median=float(np.median(values[finite])) if finite.any() else 0.0
        values[~finite]=median
        q25,q75=np.quantile(values,[.25,.75],method='linear')
        iqr=float(q75-q25)
        std=float(np.std(values,ddof=0))
        scale=iqr if iqr>EPSILON else std if std>EPSILON else 1.0
        transformed=np.arcsinh((values-median)/scale)
        mean=float(np.mean(transformed))
        transformed_std=float(np.std(transformed,ddof=0))
        output_std=transformed_std if transformed_std>EPSILON else 1.0
        _need(np.isfinite([median,iqr,std,scale,mean,output_std]).all(),'Nonfinite fit numeric statistics: '+c)
        stats.append({'column':c,'finite_fit_rows':int(finite.sum()),'median':median,'iqr':iqr,'fit_std':std,
                      'scale':float(scale),'transformed_mean':mean,'transformed_std':output_std})
    vocabularies=[]
    for c in cats:
        counts=Counter(X95[c].tolist())
        vocabularies.append(sorted(v for v,n in counts.items() if n>=MIN_CATEGORY_COUNT))
    state={'version':1,'feature_columns':list(X95.columns),'input_dtypes':{c:str(X95[c].dtype) for c in X95},
           'numeric_columns':numeric,'categorical_columns':cats,'fit_rows':int(len(X95)),
           'numeric_statistics':stats,'category_vocabularies':vocabularies,
           'cardinalities':[len(v)+1 for v in vocabularies],
           'numeric_output_columns':numeric+[c+'__nonfinite' for c in numeric],
           'numeric_output_width':144,'categorical_output_width':23,
           'min_category_count':MIN_CATEGORY_COUNT,'unknown_or_rare_category_code':0,
           'all_nonfinite_fit_median':0.0,'numeric_epsilon':EPSILON,
           'scale_fit_population':'median-imputed fit values','std_ddof':0,
           'numeric_transform':'(asinh((imputed_x-fit_median)/fit_scale)-fit_transformed_mean)/fit_transformed_std',
           'numeric_output_order':'72 standardized values, then 72 nonfinite flags in original numeric column order',
           'category_missing_policy':'Existing strings only; no new missing normalization; retained iff fit count >= 5',
           'query_used_for_state':False,'target_used_for_state':False,'clipping_applied':False}
    state['state_content_sha256']=_content_sha(state)
    return state


def encode(state,X95):
    """Apply frozen fit statistics/vocabularies, retaining input row order.

    Returns C-contiguous float32 (N,144) and int64 (N,23) arrays.
    Unknown and rare category strings map to zero; no query vocabulary is learned.
    """
    _need(state.get('version')==1,'Unsupported encoder version')
    _need(state['state_content_sha256']==_content_sha({k:v for k,v in state.items() if k!='state_content_sha256'}),
          'Encoder state was modified')
    _need(list(X95.columns)==state['feature_columns'],'Query feature order differs')
    _need({c:str(X95[c].dtype) for c in X95}==state['input_dtypes'],'Query feature dtypes differ')
    numeric=_frame(X95,state['categorical_columns'])
    _need(numeric==state['numeric_columns'],'Numeric feature order differs')
    num=np.empty((len(X95),144),dtype=np.float32)
    for j,(c,stat) in enumerate(zip(numeric,state['numeric_statistics'])):
        values=X95[c].to_numpy(dtype=np.float64,copy=True)
        missing=~np.isfinite(values)
        values[missing]=stat['median']
        with np.errstate(over='raise',invalid='raise'):
            transformed=(np.arcsinh((values-stat['median'])/stat['scale'])-stat['transformed_mean'])/stat['transformed_std']
            num[:,j]=transformed
        num[:,j+72]=missing.astype(np.float32)
    cat=np.empty((len(X95),23),dtype=np.int64)
    for j,(c,vocab) in enumerate(zip(state['categorical_columns'],state['category_vocabularies'])):
        codes={value:i+1 for i,value in enumerate(vocab)}
        cat[:,j]=X95[c].map(codes).fillna(0).to_numpy(dtype=np.int64)
        _need((cat[:,j]>=0).all() and (cat[:,j]<state['cardinalities'][j]).all(),'Category code outside vocabulary')
    _need(np.isfinite(num).all(),'Encoded numeric values must be finite; no clipping fallback')
    return num,cat


def self_check():
    """Small in-memory unknown/missing/row-order/state-roundtrip check; no real data."""
    import pickle
    idx=pd.Index([7,2,11,5,19,23,29,31,37,41,43,47])
    numeric={f'n{i}':np.arange(12,dtype=np.float32) for i in range(72)}
    numeric['n0']=np.array([0,1,2,3,4,5,6,7,8,np.nan,np.inf,-np.inf],dtype=np.float32)
    numeric['n1']=np.full(12,np.nan,dtype=np.float32)
    numeric['n2']=np.ones(12,dtype=np.float32)
    cats=[f'c{i}' for i in range(23)]
    X=pd.DataFrame({**numeric,**{c:['__MISSING__']*5+['common']*5+['rare1','rare2'] for c in cats}},index=idx)
    before=X.copy(deep=True)
    state=fit_encoder(X,cats)
    serialized=pickle.dumps(state)
    query=X.iloc[[11,0,8]].copy(deep=True)
    query.loc[query.index[0],'c0']='never_seen'
    query_before=query.copy(deep=True)
    num,cat=encode(state,query)
    restored=pickle.loads(serialized)
    num2,cat2=encode(restored,query)
    encoded_fit=encode(state,X)
    _need(num.shape==(3,144) and cat.shape==(3,23) and num.dtype==np.float32 and cat.dtype==np.int64,
          'Synthetic output shape/dtype differs')
    _need(cat[0,0]==0 and cat[0,1]==0 and cat[1,0]==1 and cat[2,0]==2 and state['cardinalities']==[3]*23,
          'Unknown/rare/retained existing missing-string encoding differs')
    _need(np.array_equal(num[:,72],np.array([1,0,0],dtype=np.float32)) and (num[:,73]==1).all()
          and (num[:,1]==0).all() and (num[:,2]==0).all(),'Numeric nonfinite/constant handling differs')
    _need(np.array_equal(num,encoded_fit[0][[11,0,8]]) and np.array_equal(num,num2) and np.array_equal(cat,cat2),
          'Row order or state roundtrip differs')
    _need(X.equals(before) and query.equals(query_before) and pickle.dumps(state)==serialized,
          'Input or fit state mutated during encoding')
    return {'status':'passed','synthetic_only':True,'fit_rows':12,'query_rows':3,
            'numeric_shape':list(num.shape),'categorical_shape':list(cat.shape),
            'checks':['fit-only unknown/rare and existing missing string','nonfinite flags/all-missing/constant numeric',
                      'row order and pickle roundtrip','fit/query/state not mutated']}


if __name__=='__main__':
    print(json.dumps(self_check(),ensure_ascii=False,indent=2))
