"""R11 fixed pure-X domain weighting; no property prices are accepted or read."""
from __future__ import annotations
import hashlib,json,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.special import expit
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,log_loss
from threadpoolctl import threadpool_limits
import joblib
ROOT=Path(__file__).resolve().parents[1]
CATS=('address_city','city_original_district','building_type','main_use','transaction_type','parking_type')
NUMS=('building_area','land_area','parking_area','rooms','living_rooms','bathrooms','floor_min','floor_max','total_floors','parking_count')
CONFIG={'version':1,'seed':20260925,'folds':5,'categorical_columns':list(CATS),'numeric_columns':list(NUMS),
 'category_min_source_fold_count':50,'reserved_categories':['M','U','R'],'value_prefix':'V:',
 'numeric_transform':'signed log1p for floor_min/max; nonnegative log1p others; invalid to missing',
 'numeric_statistics':'source train fold only; median fill then filled-source mean/std; all missing median0 scale1',
 'missing_indicators':'10 binary indicators, not standardized','query_X_used_for_weights':True,'query_labels_used':False,
 'classifier':{'C':.1,'l1_ratio':0.,'solver':'lbfgs','max_iter':1000,'tol':1e-6,'class_weight':None,'fit_intercept':True,'random_state':42},
 'classifier_loss_sample_weight':'uniform; no class balancing or subsampling','threads':4,
 'group_key':'named city/address_district/project JSON; missing road+section then address then global unresolved',
 'fold_assignment':'sha256 of R11|20260925| followed by group JSON; first 8 bytes little endian modulo5',
 'ratio':'decision_function + log(fold source-train rows / fold query-train rows)',
 'clip_ratio':[.25,4.],'normalization':'clip in log space, exp, divide by complete OOF source mean; no second clipping',
 'support_min_fit_rows':20,'support_min_fit_groups':3,
 'feasibility':{'min_region_query_support':.9,'min_structure_query_support':.9,'max_clipped_fit_fraction':.25,
  'min_row_ess_fraction':.5,'min_group_ess_ratio_to_uniform':.5,'max_group_mass_floor':.05,'max_group_mass_multiple_of_uniform':2.},
 'feasibility_limits_are_engineering_not_significance':True,'classifier_auc_is_diagnostic_not_selection_target':True,
 'auc_definition':'pooled OOF prior-corrected log_ratio; also per-fold AUC',
 'logloss_definition':'OOF log_ratio plus full-window log(nquery/nsource) as common mixture prior'}

def need(ok,message):
 if not ok:raise ValueError(message)
def value_sha(a,dtype='<f8'):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def content_sha(v):return hashlib.sha256(json.dumps(v,sort_keys=True,ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def frame_sha(X):return hashlib.sha256(pd.util.hash_pandas_object(X,index=True).to_numpy(dtype='<u8').tobytes()).hexdigest()
def norm(s):
 v=s.fillna('__MISSING__').astype(str).str.strip().reset_index(drop=True)
 return v.mask(v.isin(['','nan','None','<NA>','NaT']),'__MISSING__')
def strings(X,key):
 if key=='city_original_district':
  city,dist=norm(X.address_city),norm(X.district)
  return pd.Series([json.dumps([a,b],ensure_ascii=False,separators=(',',':')) if a!='__MISSING__' and b!='__MISSING__' else '__MISSING__' for a,b in zip(city,dist)])
 return norm(X[key])
def check_X(X):
 need(isinstance(X,pd.DataFrame) and X.columns.is_unique,'Expected feature DataFrame')
 need(not {'總價元','編號','y'}.intersection(X.columns),'Prices and row IDs cannot be input columns')
 need(set(NUMS)|{'address_city','district','address_district','project_name','address_raw','address_road','address_section','building_type','main_use','transaction_type','parking_type'}<=set(X.columns),'Missing fixed domain/group columns')
 need(X.index.is_unique,'Feature row index must be unique')
def transformed_numbers(X):
 result=[]
 for col in NUMS:
  a=X[col].to_numpy(dtype='float64',copy=True);a[~np.isfinite(a)]=np.nan
  if col in ('floor_min','floor_max'):a=np.sign(a)*np.log1p(np.abs(a))
  else:a[a<0]=np.nan;a=np.log1p(a)
  result.append(a)
 return np.column_stack(result)
def group_keys(X):
 check_X(X);fields={k:norm(X[k]).to_numpy() for k in ['address_city','address_district','project_name','address_raw','address_road','address_section']}
 result=[];counts={'named':0,'missing_road':0,'missing_address':0,'unresolved':0}
 for i in range(len(X)):
  city,district,project=(fields[k][i] for k in ['address_city','address_district','project_name'])
  if project!='__MISSING__':kind='named';key=['project',city,district,project]
  elif fields['address_road'][i]!='__MISSING__':kind='missing_road';key=[kind,city,district,fields['address_road'][i],fields['address_section'][i]]
  elif fields['address_raw'][i]!='__MISSING__':kind='missing_address';key=[kind,city,district,fields['address_raw'][i]]
  else:kind='unresolved';key=['unresolved_missing_project']
  counts[kind]+=1;result.append(json.dumps(key,ensure_ascii=False,separators=(',',':')))
 return np.asarray(result,dtype=object),counts

def assign_folds(keys):
 unique,inverse=np.unique(keys,return_inverse=True)
 values=np.array([int.from_bytes(hashlib.sha256(('R11|20260925|'+k).encode()).digest()[:8],'little')%5 for k in unique],dtype='int8')
 return values[inverse]

def fit_preprocessor(X):
 check_X(X);a=transformed_numbers(X);median=[];all_missing=[]
 for j in range(a.shape[1]):
  finite=a[:,j][np.isfinite(a[:,j])];median.append(float(np.median(finite)) if len(finite) else 0.);all_missing.append(not len(finite))
 filled=np.where(np.isnan(a),np.array(median),a);mean=filled.mean(axis=0);scale=filled.std(axis=0);scale[scale==0]=1.
 need(np.isfinite(mean).all() and np.isfinite(scale).all(),'Numeric preprocessing not finite')
 state={'version':1,'config_sha256':content_sha(CONFIG),'source_rows':len(X),'fit_X_sha256':frame_sha(X),'median':median,
  'mean':mean.tolist(),'scale':scale.tolist(),'all_missing':all_missing,'categories':{},'feature_names':[]}
 for col in CATS:
  values=strings(X,col);counts=values[values!='__MISSING__'].value_counts();frequent=sorted(counts[counts>=CONFIG['category_min_source_fold_count']].index.tolist())
  tokens=['M','U','R']+['V:'+v for v in frequent]
  state['categories'][col]={'seen':sorted(counts.index.tolist()),'frequent':frequent,'tokens':tokens}
  state['feature_names'] += ['cat:'+col+'='+v for v in tokens]
 state['feature_names'] += ['num:'+c for c in NUMS]+['missing:'+c for c in NUMS]
 state['state_sha256']=content_sha(state);return state

def encode(X,state):
 check_X(X);need(state['config_sha256']==content_sha(CONFIG),'Encoder rules changed')
 need(state['state_sha256']==content_sha({k:v for k,v in state.items() if k!='state_sha256'}),'Encoder state changed')
 cols=[];offset=0
 for col in CATS:
  cfg=state['categories'][col];seen=set(cfg['seen']);frequent=set(cfg['frequent']);tokens={v:i for i,v in enumerate(cfg['tokens'])}
  values=strings(X,col)
  code=np.fromiter((tokens['M' if v=='__MISSING__' else 'V:'+v if v in frequent else 'R' if v in seen else 'U']+offset for v in values),dtype='int32',count=len(X))
  cols.append(code);offset+=len(tokens)
 rows=np.repeat(np.arange(len(X)),len(CATS));indices=np.column_stack(cols).ravel()
 cat=sparse.csr_matrix((np.ones(len(indices)),(rows,indices)),shape=(len(X),offset))
 a=transformed_numbers(X);missing=np.isnan(a);filled=np.where(missing,np.asarray(state['median']),a)
 num=(filled-np.asarray(state['mean']))/np.asarray(state['scale'])
 need(np.isfinite(num).all(),'Nonfinite encoded numbers')
 matrix=sparse.hstack([cat,sparse.csr_matrix(num),sparse.csr_matrix(missing.astype('float64'))],format='csr')
 need(matrix.shape==(len(X),len(state['feature_names'])),'Encoder feature shape differs');return matrix

def support(X,Xq,groups):
 def keys(frame,kind):
  fields=[strings(frame,'city_original_district')] if kind=='region' else [norm(frame[c]) for c in ['building_type','main_use','transaction_type']]
  rows=list(zip(*fields));return np.array([json.dumps(list(v),ensure_ascii=False,separators=(',',':')) if '__MISSING__' not in v else '__MISSING__' for v in rows],dtype=object)
 result={}
 for kind in ['region','structure']:
  a,b=keys(X,kind),keys(Xq,kind);table=pd.DataFrame({'key':a,'group':groups});counts=table.groupby('key').size();n_groups=table.groupby('key')['group'].nunique()
  allowed=set(counts[(counts>=CONFIG['support_min_fit_rows'])&(n_groups>=CONFIG['support_min_fit_groups'])].index)-{'__MISSING__'}
  present=set(a)-{'__MISSING__'};adequate=np.array([v in allowed for v in b]);unseen=np.array([v not in present for v in b])
  result[kind]={'query_rows':len(b),'adequate_query_rows':int(adequate.sum()),'adequate_query_fraction':float(adequate.mean()),
   'unseen_or_missing_query_rows':int(unseen.sum()),'supported_fit_keys':len(allowed),'observed_fit_nonmissing_keys':len(present)}
 return result

def quantiles(a):return {str(q):float(np.quantile(a,q)) for q in [0,.01,.05,.25,.5,.75,.95,.99,1.]}
def effective_n(w):return float(np.sum(w)**2/np.sum(np.square(w)))
def weight_summary(log_ratio,groups):
 need(np.asarray(log_ratio).ndim==1 and np.isfinite(log_ratio).all(),'Invalid raw log ratio')
 lo,hi=np.log(CONFIG['clip_ratio']);clipped=np.exp(np.clip(log_ratio,lo,hi));weights=clipped/clipped.mean()
 codes,unique=pd.factorize(pd.Series(groups,dtype=object),sort=True);mass=np.bincount(codes,weights=weights);counts=np.bincount(codes).astype('float64')
 raw_q={k:float(np.exp(v)) if np.log(np.nextafter(np.float64(0),np.float64(1)))<=v<=np.log(np.finfo('float64').max) else None for k,v in quantiles(log_ratio).items()}
 summary={'raw_log_ratio_quantiles':quantiles(log_ratio),'raw_ratio_quantiles_when_representable':raw_q,
  'raw_ratio_underflow_or_overflow_rows':int(((log_ratio<np.log(np.nextafter(np.float64(0),np.float64(1))))|(log_ratio>np.log(np.finfo('float64').max))).sum()),
  'clipped_ratio_quantiles':quantiles(clipped),'clipped_ratio_mean':float(clipped.mean()),'normalized_weight_quantiles':quantiles(weights),
  'normalized_weight_mean':float(weights.mean()),'clipped_lower_rows':int((log_ratio<lo).sum()),'clipped_upper_rows':int((log_ratio>hi).sum()),
  'clipped_fraction':float(((log_ratio<lo)|(log_ratio>hi)).mean()),'row_ess':effective_n(weights),'row_ess_fraction':effective_n(weights)/len(weights),
  'groups':len(unique),'group_mass_ess':effective_n(mass),'uniform_group_mass_ess':effective_n(counts),
  'group_ess_ratio_to_uniform':effective_n(mass)/effective_n(counts),'largest_group_weight_share':float(mass.max()/mass.sum()),
  'largest_uniform_group_row_share':float(counts.max()/counts.sum())}
 return weights,clipped,summary

def feasibility_checks(summary,coverage,converged=True):
 cfg=CONFIG['feasibility'];mass_limit=max(cfg['max_group_mass_floor'],cfg['max_group_mass_multiple_of_uniform']*summary['largest_uniform_group_row_share'])
 return {'converged_all_folds':bool(converged),'query_region_support':coverage['region']['adequate_query_fraction']>=cfg['min_region_query_support'],
  'query_structure_support':coverage['structure']['adequate_query_fraction']>=cfg['min_structure_query_support'],
  'clipped_fraction_within_limit':summary['clipped_fraction']<=cfg['max_clipped_fit_fraction'],
  'row_ess_at_least_half':summary['row_ess_fraction']>=cfg['min_row_ess_fraction'],
  'group_ess_not_halved':summary['group_ess_ratio_to_uniform']>=cfg['min_group_ess_ratio_to_uniform'],
  'largest_group_mass_within_limit':summary['largest_group_weight_share']<=mass_limit}

def fit_weights(X,Xq,destination):
 check_X(X);check_X(Xq);destination=Path(destination);need(destination.is_dir(),'Caller must create fresh safe staging directory')
 need(len(X)>0 and len(Xq)>0,'Empty domain');source_sha=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
 gs,counts_s=group_keys(X);gq,counts_q=group_keys(Xq);fs,fq=assign_folds(gs),assign_folds(gq)
 allkeys=np.r_[gs,gq];allfolds=np.r_[fs,fq];lookup={}
 for key,fold in zip(allkeys,allfolds):
  need(key not in lookup or lookup[key]==int(fold),'Group leaks across folds');lookup[key]=int(fold)
 log_ratio=np.full(len(X),np.nan);query_log_ratio=np.full(len(Xq),np.nan);ps=np.full(len(X),np.nan);pq=np.full(len(Xq),np.nan);fold_reports=[]
 for fold in range(5):
  si,qi=np.flatnonzero(fs!=fold),np.flatnonzero(fq!=fold);sv,qv=np.flatnonzero(fs==fold),np.flatnonzero(fq==fold)
  need(all(len(a)>0 for a in [si,qi,sv,qv]),'A group fold lacks a source/query train/heldout domain')
  held=set(allkeys[allfolds==fold]);train=set(allkeys[allfolds!=fold]);need(not held&train,'Heldout group in classifier training')
  state=fit_preprocessor(X.iloc[si].reset_index(drop=True));a=encode(X.iloc[si].reset_index(drop=True),state);b=encode(Xq.iloc[qi].reset_index(drop=True),state)
  model=LogisticRegression(**CONFIG['classifier']);D=np.r_[np.zeros(len(si),dtype='int8'),np.ones(len(qi),dtype='int8')]
  with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=4):
   warnings.simplefilter('always');model.fit(sparse.vstack([a,b],format='csr'),D)
  need(not any(issubclass(w.category,ConvergenceWarning) for w in caught),'Fixed domain classifier did not converge; do not retune')
  need(np.array_equal(model.classes_,[0,1]) and np.isfinite(model.coef_).all() and np.isfinite(model.intercept_).all(),'Invalid classifier state')
  held_source=encode(X.iloc[sv].reset_index(drop=True),state);held_query=encode(Xq.iloc[qv].reset_index(drop=True),state)
  ls=np.asarray(model.decision_function(held_source),dtype='float64');lq=np.asarray(model.decision_function(held_query),dtype='float64')
  need(np.isfinite(ls).all() and np.isfinite(lq).all(),'Nonfinite domain logits')
  prior=float(np.log(len(si)/len(qi)));log_ratio[sv]=ls+prior;query_log_ratio[qv]=lq+prior;ps[sv]=expit(ls);pq[qv]=expit(lq)
  folder=destination/('fold_'+str(fold));folder.mkdir();joblib.dump(model,folder/'classifier.joblib')
  (folder/'preprocessor.json').write_text(json.dumps(state,ensure_ascii=False,sort_keys=True,indent=2)+'\n',encoding='utf-8')
  reloaded=joblib.load(folder/'classifier.joblib');need(np.allclose(reloaded.decision_function(held_source),ls,rtol=1e-12,atol=1e-12) and np.allclose(reloaded.decision_function(held_query),lq,rtol=1e-12,atol=1e-12),'Saved classifier replay differs')
  report={'fold':fold,'source_train_rows':len(si),'query_train_rows':len(qi),'source_heldout_rows':len(sv),'query_heldout_rows':len(qv),
   'log_prior_correction':prior,'source_train_indices_sha256':value_sha(si,'<i8'),'query_train_indices_sha256':value_sha(qi,'<i8'),
   'source_heldout_indices_sha256':value_sha(sv,'<i8'),'query_heldout_indices_sha256':value_sha(qv,'<i8'),
   'preprocessor_source_train_only':True,'preprocessor_source_X_sha256':state['fit_X_sha256'],'heldout_groups_excluded_from_both_domains':True,
   'heldout_auc':float(roc_auc_score(np.r_[np.zeros(len(sv)),np.ones(len(qv))],np.r_[ls,lq])),
   'feature_count':len(state['feature_names']),'iterations':int(model.n_iter_[0]),'converged':True,'saved_classifier_all_heldout_replay_verified':True,
   'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in caught]}
  fold_reports.append(report);print(json.dumps({'event':'domain_fold_completed','fold':fold,'iterations':report['iterations']}),flush=True)
 need(np.isfinite(log_ratio).all() and np.isfinite(query_log_ratio).all() and np.isfinite(ps).all() and np.isfinite(pq).all(),'Incomplete OOF prediction')
 weights,clipped,summary=weight_summary(log_ratio,gs);coverage=support(X,Xq,gs);checks=feasibility_checks(summary,coverage)
 D=np.r_[np.zeros(len(X)),np.ones(len(Xq))];all_log_ratio=np.r_[log_ratio,query_log_ratio];prob=expit(all_log_ratio+np.log(len(Xq)/len(X)))
 city_source=norm(X.address_city);city_query=norm(Xq.address_city);city_summary={}
 for city in sorted(set(city_source)|set(city_query)):
  a=city_source.eq(city).to_numpy();b=city_query.eq(city).to_numpy()
  city_summary[city]={'source_rows':int(a.sum()),'source_unweighted_share':float(a.mean()),'source_weight_share':float(weights[a].sum()/weights.sum()),'query_rows':int(b.sum()),'query_share':float(b.mean())}
 arrays={'fit_folds':fs,'query_folds':fq,'fit_oof_probability':ps,'query_oof_probability':pq,'fit_raw_log_ratio':log_ratio,'query_raw_log_ratio':query_log_ratio,'fit_clipped_ratio':clipped,'fit_weights':weights}
 for name,a in arrays.items():np.save(destination/(name+'.npy'),a,allow_pickle=False)
 need(source_sha==hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'Weight source changed during preparation')
 return {'config':CONFIG,'config_sha256':content_sha(CONFIG),'fit_rows':len(X),'query_rows':len(Xq),'fit_X_sha256':frame_sha(X),'query_X_sha256':frame_sha(Xq),
  'fold_reports':fold_reports,'weight_summary':summary,'coverage':coverage,'feasibility_checks':checks,'feasibility_passed':all(checks.values()),
  'diagnostic_domain_oof_auc_prior_corrected':float(roc_auc_score(D,all_log_ratio)),'diagnostic_domain_oof_log_loss_common_prior':float(log_loss(D,prob)),
  'city_mass_summary':city_summary,
  'fit_group_fallback_counts':counts_s,'query_group_fallback_counts':counts_q,'fit_weight_values_sha256':value_sha(weights),
  'query_X_used_for_weights':True,'query_y_or_price_cache_used':False,'classifier_models_fitted':5,'property_price_models_fitted':0}

def self_check():
 checks=[]
 def passed(name):checks.append(name)
 X=pd.DataFrame({c:np.full(100,2.,dtype='float64') for c in NUMS})
 for c in ['address_city','district','address_district','project_name','address_raw','address_road','address_section','building_type','main_use','transaction_type','parking_type']:X[c]='a'
 X.loc[:4,'building_area']=np.nan;X.loc[5,'land_area']=-1;X.loc[6,'floor_min']=-2;original=X.copy(deep=True)
 state=fit_preprocessor(X);matrix=encode(X,state)
 need(X.equals(original),'Feature transformation mutated input');passed('input_preserved')
 need(np.isnan(transformed_numbers(X)[5,1]) and transformed_numbers(X)[6,6]<0,'Invalid areas or negative floor semantics');passed('numeric_missing_signed_floor')
 need(state['median'][0]==float(np.log1p(2)),'Median must follow numeric transform');passed('transformed_median')
 missing=matrix[:,-10:].toarray();need(missing[0,0]==1 and missing[5,1]==1 and set(np.unique(missing))<={0.,1.},'Missing indicators scaled or wrong');passed('missing_flags_unscaled')
 q=X.iloc[:2].copy();q['building_type']=['new','__MISSING__'];encoded=encode(q,state);offset=sum(len(state['categories'][c]['tokens']) for c in CATS[:2]);need(encoded[0,offset+1]==1 and encoded[1,offset]==1,'Unknown/missing categories');passed('unknown_missing_reserved')
 tokens=state['categories']['address_city']['tokens'];need(tokens[:3]==['M','U','R'] and 'V:a' in tokens,'Reserved token collision');passed('token_schema')
 changed=X.copy();changed['trade_day_index']=np.arange(100);changed['remark_related_party']=1.;need((encode(changed,state)!=matrix).nnz==0,'Excluded columns changed domain representation');passed('excluded_date_remark_columns')
 bad=X.copy();bad['總價元']=1.
 try:fit_preprocessor(bad)
 except ValueError:passed('target_input_rejected')
 else:raise AssertionError('Target was allowed')
 gx,_=group_keys(X);gq,_=group_keys(X.iloc[:8]);need(np.array_equal(assign_folds(gx)[:8],assign_folds(gq)),'Same group across domain folds');passed('same_group_same_fold')
 blank=X.iloc[:3].copy();blank['project_name']='__MISSING__';blank['address_road']=['road','__MISSING__','__MISSING__'];blank['address_raw']=['different','address','__MISSING__'];keys,counts=group_keys(blank)
 need(counts=={'named':0,'missing_road':1,'missing_address':1,'unresolved':1},'Missing group policy');passed('missing_group_proxy_order')
 w,c,r=weight_summary(np.log(np.array([.1,.25,1.,4.,10.])),np.array(['a','a','b','c','d']))
 need(np.allclose(c,[.25,.25,1,4,4]) and abs(w.mean()-1)<1e-12 and r['clipped_lower_rows']==1 and r['clipped_upper_rows']==1,'Clip then normalize');passed('clipping_normalization')
 need(np.allclose(w,c/c.mean()),'Unexpected second clipping');passed('no_second_clip')
 wx,_,_=weight_summary(np.array([-1000.,1000.]),np.array(['a','b']));need(np.isfinite(wx).all() and np.all(wx>0),'Extreme log odds overflow');passed('extreme_logits_safe')
 need(effective_n(np.ones(7))==7.,'Uniform ESS');passed('uniform_effective_size')
 common=sparse.csr_matrix(np.zeros((300,2)));d=np.r_[np.zeros(200),np.ones(100)]
 with threadpool_limits(limits=4):m=LogisticRegression(**CONFIG['classifier']).fit(common,d)
 need(abs(float(m.decision_function(common[:1])[0])+np.log(2))<1e-4,'Empirical prior correction identity');passed('constant_Z_prior_correction_synthetic_classifier')
 return {'passed':True,'checks':len(checks),'names':checks,'real_data_read':False,'property_price_model_trained':False}
