"""R19 one-pass target-free effective-area audit; saved before its real X run.

Run --self-check first, then --run once. No raw-price cache, model, prediction,
valid dataset, training or scoring is loaded. Only three frozen int64 row maps
are decoded, alongside train/test feature caches and source/contract metadata.
"""
from __future__ import annotations
import argparse, ast, hashlib, json, sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round19_research/area_audit.json'
SNAPSHOT = 'runs/v3/snapshot_manifest.json'
SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
PLAN = 'outputs/round6/candidate_plan.json'
MANIFEST = 'outputs/round6/geography/development/manifest.json'
TRAIN = 'data_cache/train_features.pkl'
TEST = 'data_cache/test_features.pkl'
SOURCE_INDEX = 'outputs/round6/geography/development/train_source_cache_indices.npy'
FIT_INDEX = 'outputs/round6/models/development/geography_catboost/fit_indices.npy'
QUERY_INDEX = 'outputs/round6/models/development/geography_catboost/query_indices.npy'
SOURCE_FILES = ('src/round19_area_audit.py', 'src/train_catboost.py', 'src/model_common.py',
                'src/house_features.py', 'src/round6_geo.py', 'src/round7_geo_unit.py',
                'src/round6_common.py', 'src/round6_prepare_geo.py')
EVIDENCE_FILES = ('reports/round18_method_audit.md',)
RULES = {
    'effective_area': 'finite positive building_area else finite positive parking_area else 1',
    'target': 'y_TWD / (A * 1e4)', 'sample_weight': 'A / 100',
    'inverse': 'max(z * A * 1e4, 0)', 'dtype': 'float64',
    'fixed_area_bands': ['A < 15', '15 <= A <= 1500', 'A > 1500'],
    'fixed_weight_greater_than': [1, 10, 100],
    'quantiles': [0, 0.01, 0.5, 0.99, 1],
    'thresholds_are_diagnostics_only': True, 'clipping_filtering_or_selection': False,
    'development_dates': ['2022-08-11', '2023-08-10'],
}

def need(condition, message):
    if not condition: raise ValueError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def pure_function(name, filename, globals_):
    tree=ast.parse((ROOT/filename).read_text(encoding='utf-8-sig'))
    matches=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name]
    need(len(matches)==1 and not matches[0].decorator_list,'Unexpected source function')
    namespace=dict(globals_)
    exec(compile(ast.Module(body=matches,type_ignores=[]),filename,'exec'),namespace)
    return namespace[name]

def area_parts(X):
    b=X['building_area'].to_numpy(dtype='float64')
    p=X['parking_area'].to_numpy(dtype='float64')
    use_b=np.isfinite(b)&(b>0);use_p=~use_b&np.isfinite(p)&(p>0)
    area=np.where(use_b,b,np.where(use_p,p,1.0)).astype('float64')
    with np.errstate(over='raise',invalid='raise',divide='raise'):
        try:factor=area*1e4;weight=area/100.0
        except FloatingPointError as exc:raise ValueError('Area factor/weight overflow') from exc
    need(np.isfinite(area).all() and (area>0).all(),'Effective area is invalid')
    need(np.isfinite(factor).all() and (factor>0).all(),'Inverse factor is invalid')
    need(np.isfinite(weight).all() and (weight>0).all(),'Area weight is invalid')
    return area,weight,factor,{'building':use_b,'parking_fallback':use_p,'one_fallback':~use_b&~use_p}

def check_index(a,size,name):
    need(a.ndim==1 and a.dtype==np.dtype('int64') and len(a)>0,name+' must be nonempty int64 vector')
    need((a>=0).all() and (a<size).all() and np.all(np.diff(a)>0),name+' range/order/uniqueness differs')

def summaries(X,r1):
    area,weight,factor,masks=area_parts(X)
    prices_none,r1weights,r1factor=r1(X,None,'unit')
    need(prices_none is None and np.array_equal(weight,r1weights) and np.array_equal(factor,r1factor),
         'Exact R1 transform(prices=None) equivalence failed')
    quantiles=lambda a:{str(q):float(np.quantile(a,q)) for q in RULES['quantiles']}
    raw={}
    for col in ('building_area','parking_area'):
        a=X[col].to_numpy(dtype='float64')
        raw[col]={'nonfinite':int((~np.isfinite(a)).sum()),'negative':int((a<0).sum()),'zero':int((a==0).sum())}
    return {'rows':len(X),'effective_area_sources':{k:int(m.sum()) for k,m in masks.items()},
        'effective_area_quantiles_m2':quantiles(area),
        'area_bands':{'below_15':int((area<15).sum()),'15_through_1500':int(((area>=15)&(area<=1500)).sum()),'above_1500':int((area>1500).sum())},
        'weight_mean':float(weight.mean()),'weight_min':float(weight.min()),'weight_max':float(weight.max()),
        'weight_greater_than':{str(t):int((weight>t).sum()) for t in RULES['fixed_weight_greater_than']},
        'factor_min':float(factor.min()),'factor_max':float(factor.max()),
        'effective_area_nonpositive_or_nonfinite':int(((area<=0)|~np.isfinite(area)).sum()),
        'weight_nonpositive_or_nonfinite':int(((weight<=0)|~np.isfinite(weight)).sum()),
        'factor_nonpositive_or_nonfinite':int(((factor<=0)|~np.isfinite(factor)).sum()),
        'raw_area_quality':raw,
        'fallback_transaction_types':{k:{str(c):int(n) for c,n in X.loc[m,'transaction_type'].value_counts(dropna=False).items()} for k,m in masks.items() if k!='building'},
        'r1_transform_prices_None_exact':True,
        'effective_area_values_sha256':hashlib.sha256(area.astype('<f8').tobytes()).hexdigest(),
        'weight_values_sha256':hashlib.sha256(weight.astype('<f8').tobytes()).hexdigest()}

def self_check():
    checks=[]
    def ok(name,value):need(value,name);checks.append(name)
    def rejects(name,f):
        try:f()
        except (ValueError,FloatingPointError):checks.append(name)
        else:raise AssertionError(name+' accepted')
    X=pd.DataFrame({'building_area':[100.,0.,np.nan,np.inf,-5.,-np.inf,0.,float(np.finfo('float32').max)],
                    'parking_area':[9.,25.,0.,16.,0.,np.nan,np.inf,10.]})
    before=X.copy(deep=True);a,w,f,m=area_parts(X)
    ok('finite_positive_building_then_parking_then_one',np.array_equal(a,[100.,25.,1.,16.,1.,1.,1.,float(np.finfo('float32').max)]))
    ok('source_masks_partition_each_row',np.all(sum(x.astype(int) for x in m.values())==1))
    r1=pure_function('transform','src/train_catboost.py',{'np':np})
    y,wr,fr=r1(X,None,'unit')
    ok('actual_R1_function_prices_None_exact',y is None and np.array_equal(w,wr) and np.array_equal(f,fr))
    ok('X_unchanged_and_float64_outputs',X.equals(before) and all(v.dtype==np.float64 for v in (a,w,f)))
    rejects('factor_overflow_fails',lambda:area_parts(pd.DataFrame({'building_area':[np.finfo('float64').max],'parking_area':[0.]})))
    check_index(np.array([0,2],dtype='int64'),3,'synthetic');checks.append('valid_monotone_row_map')
    for label,values in [('duplicate',[0,0]),('negative',[-1,0]),('out_of_range',[0,3]),('reversed',[2,0])]:
        rejects('index_'+label+'_rejected',lambda v=values:check_index(np.array(v,dtype='int64'),3,'synthetic'))
    rejects('index_float_dtype_rejected',lambda:check_index(np.array([0.,1.]),3,'synthetic'))
    temporal=pure_function('temporal_indices','src/model_common.py',{'np':np,'pd':pd})
    start=(pd.Timestamp(RULES['development_dates'][0])-pd.Timestamp('2000-01-01')).days
    end=(pd.Timestamp(RULES['development_dates'][1])-pd.Timestamp('2000-01-01')).days
    fi,qi=temporal({'X_train':pd.DataFrame({'trade_day_index':[start-1,start,end,end+1,np.nan]})},*RULES['development_dates'])
    ok('original_date_boundaries_and_invalid_exclusion',np.array_equal(fi,[0]) and np.array_equal(qi,[1,2]))
    return checks

def run():
    receipt=OUT.with_suffix('.sha256')
    need(not OUT.exists() and not receipt.exists(),'Refusing audit overwrite/partial output')
    checks=self_check()
    source_before={p:sha(ROOT/p) for p in SOURCE_FILES}
    evidence={p:sha(ROOT/p) for p in EVIDENCE_FILES}
    input_names=[TRAIN,TEST,SOURCE_INDEX,FIT_INDEX,QUERY_INDEX,PLAN,MANIFEST,SNAPSHOT]
    input_names += [str(Path(p).with_suffix('.sha256')).replace('\\','/') for p in (PLAN,MANIFEST,SNAPSHOT)]
    inputs={p:{'path':p,'sha256':sha(ROOT/p),'bytes':(ROOT/p).stat().st_size} for p in input_names}
    def locked(name):
        path=ROOT/name
        need(path.with_suffix('.sha256').read_text(encoding='ascii').strip()==inputs[name]['sha256'],'Receipt differs: '+name)
        return json.loads(path.read_text(encoding='utf-8-sig'))
    need(inputs[SNAPSHOT]['sha256']==SNAPSHOT_SHA,'Released v3 source authority changed')
    snapshot=locked(SNAPSHOT)
    known={k:v['original_sha256'] for k,v in snapshot['copied_files'].items()}
    for v in snapshot['external_dependencies']:
        name=v['original_path'];digest=v['sha256']
        need(name not in known or known[name]==digest,'Conflicting released identity');known[name]=digest
    for name in (TRAIN,TEST,SOURCE_INDEX,FIT_INDEX,QUERY_INDEX,PLAN,MANIFEST):
        need(known.get(name)==inputs[name]['sha256'],'Input is not the released R6 identity: '+name)
    plan=locked(PLAN);manifest=locked(MANIFEST)
    need(plan['round']==6 and plan['splits']['development']==RULES['development_dates'],'Wrong original split')
    need(manifest['stage']=='development' and manifest['index_files_sha256'][SOURCE_INDEX]==inputs[SOURCE_INDEX]['sha256'],'Canonical mapping not manifest-bound')
    source_idx=np.load(ROOT/SOURCE_INDEX,allow_pickle=False)
    fi=np.load(ROOT/FIT_INDEX,allow_pickle=False);qi=np.load(ROOT/QUERY_INDEX,allow_pickle=False)
    train=pd.read_pickle(ROOT/TRAIN);test=pd.read_pickle(ROOT/TEST)
    for X in (train,test):
        need(len(X.columns)==87 and X.columns.is_unique and X.index.is_unique,'Wrong original X schema')
        need(not {'總價元','編號','y','target'}&set(X.columns),'Forbidden target or ID in X')
        need(sum(not pd.api.types.is_numeric_dtype(X[c]) for c in X)==21,'Wrong category count')
    need(len(train)==455784 and len(test)==47073,'Provided feature-cache row count differs')
    check_index(source_idx,len(train),'canonical source cache map')
    need(len(source_idx)==455255,'Expected original 529-row dedup mapping')
    canonical=train.iloc[source_idx].reset_index(drop=True)
    check_index(fi,len(canonical),'fit indices');check_index(qi,len(canonical),'query indices')
    need(not np.intersect1d(fi,qi).size,'Fit/query overlap')
    contract=plan['split_contract']['development']
    for role,indices in [('fit',fi),('query',qi)]:
        need(len(indices)==contract[role+'_rows'] and hashlib.sha256(indices.astype('<i8').tobytes()).hexdigest()==contract[role+'_indices_content_sha256'],'Saved split identity differs')
    temporal=pure_function('temporal_indices','src/model_common.py',{'np':np,'pd':pd})
    expected_fit,expected_query=temporal({'X_train':canonical},*RULES['development_dates'])
    need(np.array_equal(fi,expected_fit) and np.array_equal(qi,expected_query),'Date-to-frozen-index mapping differs')
    r1=pure_function('transform','src/train_catboost.py',{'np':np})
    views={'train_cache_before_529_dedup':train,'canonical_train':canonical,
           'development_fit':canonical.iloc[fi], 'development_query':canonical.iloc[qi], 'test_cache':test}
    data={name:summaries(X,r1) for name,X in views.items()}
    need(source_before=={p:sha(ROOT/p) for p in SOURCE_FILES},'Source changed during audit')
    need(evidence=={p:sha(ROOT/p) for p in EVIDENCE_FILES},'Method evidence changed during audit')
    need(all(sha(ROOT/p)==item['sha256'] and (ROOT/p).stat().st_size==item['bytes'] for p,item in inputs.items()),'Input changed during audit')
    result={'round':19,'status':'target_free_effective_area_support_audit','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'source_sha256':source_before,'evidence_sha256':evidence,'inputs':inputs,'rules':RULES,
        'self_checks':checks,'splits':data,'alignment':{'train_cache_rows':len(train),'canonical_rows':len(canonical),
         'original_duplicate_rows_removed':len(train)-len(canonical),'fit_rows':len(fi),'query_rows':len(qi),
         'original_saved_mapping_reused':True,'original_dedup_historically_included_target':True,
         'target_dedup_recomputed_here':False,'indices_and_dates_match_released_R6':True},
        'read_scope':{'target_cells_decoded':0,'train_test_target_free_caches_only':True,
         'raw_price_cache_or_Excel_loaded':False,'valid_dataset_loaded':False,
         'three_int64_row_maps_loaded':True,'model_or_prediction_loaded':False,'MAE_computed':False,
         'candidate_fitted':False,'test_X_distribution_observed':True,'query_test_weights_descriptive_not_training':True},
        'prices_None_R1_transform_used':True,'feature_values_modified':False,
        'nonnegative_finite_label_and_prediction_checks_deferred_to_actual_fit':True,
        'runtime_versions':{'python':sys.version.split()[0],'numpy':np.__version__,'pandas':pd.__version__}}
    text=(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
    OUT.parent.mkdir(parents=True,exist_ok=True)
    with OUT.open('xb') as f:f.write(text)
    with receipt.open('x',encoding='ascii') as f:f.write(sha(OUT)+'\n')
    print(json.dumps({'status':'completed','path':OUT.relative_to(ROOT).as_posix(),'sha256':sha(OUT),
          'source_sha256':source_before['src/round19_area_audit.py'],'rows':{k:v['rows'] for k,v in data.items()},
          'self_check_count':len(checks),'splits':data},ensure_ascii=False),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--self-check',action='store_true');mode.add_argument('--run',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps({'status':'passed','checks':self_check(),'real_X_read':False},ensure_ascii=False))
    else:run()
