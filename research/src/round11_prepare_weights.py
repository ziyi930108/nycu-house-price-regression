"""Prepare fixed R11 importance weights using feature caches only; never load price caches."""
from __future__ import annotations
import argparse,hashlib,json,platform,uuid
from datetime import datetime,timezone
from importlib.metadata import version
from pathlib import Path
import numpy as np
import pandas as pd
import round11_domain_weights as domain
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round11'
SPLITS={'development':('2022-08-11','2023-08-10'),'confirmation':('2023-08-11','2024-08-10'),'auxiliary':('2021-08-11','2022-08-10')}
SNAPSHOT_SHA='77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
DOCS=['reports/round11_covariate_shift_feasibility.md','reports/round11_weighting_design_review.md','reports/round11_domain_plan.md','reports/round11_X_inventory.md','reports/round11_code_review.md','outputs/round11/research/inventory.json']

def need(ok,message):
 if not ok:raise RuntimeError(message)

def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
 return h.hexdigest()

def rel(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def locked(path):
 path=Path(path);need(path.with_suffix('.sha256').read_text(encoding='ascii').strip()==sha(path),'Receipt mismatch: '+str(path));return read(path)

def write_locked(path,value):
 path=Path(path);need(not path.exists() and not path.with_suffix('.sha256').exists(),'Refusing overwrite: '+str(path));path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('x',encoding='utf-8') as f:f.write(json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n')
 with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')

def verify(mapping):
 for name,digest in mapping.items():
  path=(ROOT/name).resolve();need(path.is_relative_to(ROOT.resolve()) and path.is_file(),'Unsafe/missing dependency '+name);need(sha(path)==digest,'Changed dependency '+name)

def runtime():return {'python':platform.python_version(),**{name:version(name) for name in ['numpy','pandas','scipy','scikit-learn','joblib','threadpoolctl']}}
def sources():return {name:sha(ROOT/name) for name in ['src/round11_domain_weights.py','src/round11_prepare_weights.py']}
def idx_paths(stage):
 folder=ROOT/'outputs/round6/models'/stage/'geography_catboost'
 return folder/'fit_indices.npy',folder/'query_indices.npy'

def input_paths():
 paths=['runs/v3/snapshot_manifest.json','reports/feature_schema.json','outputs/round6/candidate_plan.json']
 paths += ['data_cache/'+s+'_features.pkl' for s in ['train','valid','test']]
 for stage in [*SPLITS,'final']:
  paths += [rel(p) for p in idx_paths(stage)]
  paths += ['outputs/round6/geography/'+stage+'/'+s+'_source_cache_indices.npy' for s in ['train','valid','test','full']]
 return paths

def plan_create():
 snapshot=ROOT/'runs/v3/snapshot_manifest.json';need(sha(snapshot)==SNAPSHOT_SHA,'Historical v3 snapshot changed');old=read(snapshot)
 historical={name:item['original_sha256'] for name,item in old['copied_files'].items()}
 historical.update({item['original_path']:item['sha256'] for item in old['external_dependencies']})
 historical[rel(snapshot)]=SNAPSHOT_SHA
 inputs={}
 for name in input_paths():
  need(name in historical,'Input missing historical evidence: '+name);need(sha(ROOT/name)==historical[name],'Historical input changed: '+name);inputs[name]=historical[name]
 verify(inputs)
 for name in DOCS:inputs[name]=sha(ROOT/name)
 p={'round':11,'kind':'pure_X_domain_weight_plan','created_at_utc':datetime.now(timezone.utc).isoformat(),'config':domain.CONFIG,'config_sha256':domain.content_sha(domain.CONFIG),
  'source_sha256':sources(),'input_sha256':inputs,'runtime_versions':runtime(),'splits':{k:list(v) for k,v in SPLITS.items()},'reference_snapshot_sha256':SNAPSHOT_SHA,
  'domain_query_labels_read':False,'real_domain_fits_before_plan':0,'price_models_before_plan':0,'external_sale_prices_used':False,
  'later_stage_preparation_requires_this_round_price_development_pass':True,'classifier_auc_used_for_tuning':False,
  'stop_on_any_feasibility_failure':True,'historically_untouched':False,'one_configuration_no_parameter_search':True}
 verify(p['source_sha256']);verify(p['input_sha256']);need(p['runtime_versions']==runtime(),'Runtime changed before plan freeze')
 write_locked(OUT/'domain_plan.json',p);print(json.dumps({'domain_plan_sha256':sha(OUT/'domain_plan.json'),'frozen':True}),flush=True)

def plan():
 p=locked(OUT/'domain_plan.json');need(p['round']==11 and p['kind']=='pure_X_domain_weight_plan','Wrong plan');need(p['config']==domain.CONFIG and p['config_sha256']==domain.content_sha(domain.CONFIG),'Domain config changed')
 need(p['source_sha256']==sources(),'Source changed');verify(p['source_sha256']);verify(p['input_sha256']);need(p['runtime_versions']==runtime(),'Runtime changed');return p

def authorize(stage):
 need(stage in [*SPLITS,'final'],'Unknown phase');p=plan()
 if stage!='development':
  frozen=locked(OUT/'frozen_candidate.json');priceplan=locked(OUT/'candidate_plan.json');result=locked(OUT/'evaluation/development/results.json')
  need(frozen['passed'] is True and frozen['later_stages_allowed'] is True and frozen['selected_candidate']=='importance_geo_blend','R11 development did not authorize later stages')
  need(frozen['plan_sha256']==sha(OUT/'candidate_plan.json') and frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'R11 development evidence differs')
  need(result['decision']['passed'] is True and result['decision']['selected_candidate']=='importance_geo_blend','R11 development result failed')
  verify(frozen['source_sha256']);need(priceplan['round']==11 and priceplan['domain_plan_sha256']==sha(OUT/'domain_plan.json'),'Wrong price/domain plan')
  if stage=='final':
   accepted=locked(OUT/'accepted_candidate.json');need(accepted['accepted'] is True,'Final preparation requires R11 acceptance')
   need(accepted['round']==11 and accepted['selected_candidate']=='importance_geo_blend','Wrong accepted candidate')
   need(accepted['plan_sha256']==sha(OUT/'candidate_plan.json') and accepted['frozen_candidate_sha256']==sha(OUT/'frozen_candidate.json'),'Acceptance plan/freeze identity differs')
   for s in SPLITS:
    r=locked(OUT/'evaluation'/s/'results.json');need(r['decision']['passed'] is True,'Final requires all three R11 passes');need(accepted['stage_result_sha256'][s]==sha(OUT/'evaluation'/s/'results.json'),'Acceptance result mismatch')
 return p

def array(path):
 a=np.load(path,allow_pickle=False);need(a.ndim==1 and np.issubdtype(a.dtype,np.integer),'Invalid index array');return a.astype('int64',copy=False)

def partition_X(stage):
 """No imports of shared target-capable loaders; use immutable saved row maps."""
 schema=read(ROOT/'reports/feature_schema.json');folder=ROOT/'outputs/round6/geography'/stage
 def cache(name):
  frame=pd.read_pickle(ROOT/'data_cache'/(name+'_features.pkl'));need(list(frame)==schema['columns'] and frame.shape[1]==87 and len(frame)==schema['splits'][name]['rows'],'Feature cache schema differs');domain.check_X(frame);return frame
 def mapped(frame,name):
  ix=array(folder/(name+'_source_cache_indices.npy'));need(len(np.unique(ix))==len(ix) and np.all((ix>=0)&(ix<len(frame))),'Invalid saved row map');return frame.iloc[ix].reset_index(drop=True)
 train=mapped(cache('train'),'train');need(len(train)==455255,'Canonical train count differs')
 fi,qi=[array(p) for p in idx_paths(stage)]
 old=read(ROOT/'outputs/round6/candidate_plan.json')
 if stage=='final':
  full=mapped(pd.concat([train,mapped(cache('valid'),'valid')],ignore_index=True),'full');test=mapped(cache('test'),'test')
  need(np.array_equal(fi,np.arange(len(full))) and np.array_equal(qi,np.arange(len(test))),'Final saved indices differ');X,Xq=full,test
  need(len(X)==467221 and len(Xq)==47073,'Final counts differ')
 else:
  contract=old['split_contract'][stage]
  for label,ix in [('fit',fi),('query',qi)]:
   need(len(ix)==contract[label+'_rows'] and domain.value_sha(ix,'<i8')==contract[label+'_indices_content_sha256'],'Saved annual indices differ')
   need(len(np.unique(ix))==len(ix) and np.all((ix>=0)&(ix<len(train))),'Invalid annual indices')
  need(not np.intersect1d(fi,qi).size,'Fit/query overlap');X=train.iloc[fi].reset_index(drop=True);Xq=train.iloc[qi].reset_index(drop=True)
  start,end=[(pd.Timestamp(d)-pd.Timestamp('2000-01-01')).days for d in SPLITS[stage]];days=train.trade_day_index.to_numpy()
  need(np.array_equal(fi,np.flatnonzero(days<start)) and np.array_equal(qi,np.flatnonzero((days>=start)&(days<=end))),'Date-derived split differs')
 return X,Xq,fi,qi

def prepare(stage):
 p=authorize(stage);dest=OUT/'weights'/stage;need(not dest.exists(),'Preparation already exists');staging=OUT/'weights'/('_'+stage+'_'+uuid.uuid4().hex);staging.mkdir(parents=True)
 X,Xq,fi,qi=partition_X(stage);before=(domain.frame_sha(X),domain.frame_sha(Xq));print(json.dumps({'event':'pure_X_partition_ready','stage':stage,'fit_rows':len(X),'query_rows':len(Xq)}),flush=True)
 result=domain.fit_weights(X,Xq,staging)
 need(before==(domain.frame_sha(X),domain.frame_sha(Xq)),'Frames mutated during preparation')
 result.update({'round':11,'stage':stage,'created_at_utc':datetime.now(timezone.utc).isoformat(),'domain_plan_sha256':sha(OUT/'domain_plan.json'),
  'source_sha256':p['source_sha256'],'input_sha256':p['input_sha256'],'runtime_versions':runtime(),
  'fit_indices_content_sha256':domain.value_sha(fi,'<i8'),'query_indices_content_sha256':domain.value_sha(qi,'<i8'),
  'original_feature_columns':list(X),'fit_feature_dtypes':{k:str(v) for k,v in X.dtypes.items()},'original_features_mutated':False,
  'row_maps_reused_without_deserializing_price_cache':True,'historically_target_containing_dedup_used_for_alignment':True})
 result['artifact_sha256']={rel(dest/f.relative_to(staging)):sha(f) for f in staging.rglob('*') if f.is_file()}
 authorize(stage);write_locked(staging/'manifest.json',result);need(not dest.exists(),'Destination concurrently created');staging.rename(dest)
 verify(result['artifact_sha256']);print(json.dumps({'stage':stage,'feasibility_passed':result['feasibility_passed'],'checks':result['feasibility_checks'],'manifest_sha256':sha(dest/'manifest.json')}),flush=True)

def load_prepared(stage,X,Xq,fi,qi):
 p=authorize(stage);folder=OUT/'weights'/stage;m=locked(folder/'manifest.json')
 need(m['round']==11 and m['stage']==stage and m['domain_plan_sha256']==sha(OUT/'domain_plan.json'),'Prepared weights belong to different plan/phase')
 need(m['config']==domain.CONFIG and m['source_sha256']==p['source_sha256'] and m['input_sha256']==p['input_sha256'] and m['runtime_versions']==runtime(),'Preparation provenance differs')
 verify(m['source_sha256']);verify(m['input_sha256']);verify(m['artifact_sha256'])
 need(m['feasibility_passed'] is True and all(m['feasibility_checks'].values()),'Pure-X feasibility failed: price fitting prohibited')
 need(m['feasibility_checks']==domain.feasibility_checks(m['weight_summary'],m['coverage']),'Feasibility evidence differs')
 need(m['query_X_used_for_weights'] is True and m['query_y_or_price_cache_used'] is False,'Invalid preparation data access')
 for key,value in [('fit_X_sha256',domain.frame_sha(X)),('query_X_sha256',domain.frame_sha(Xq)),('fit_indices_content_sha256',domain.value_sha(fi,'<i8')),('query_indices_content_sha256',domain.value_sha(qi,'<i8'))]:need(m[key]==value,'Prepared input differs: '+key)
 need(list(X)==m['original_feature_columns'] and {k:str(v) for k,v in X.dtypes.items()}==m['fit_feature_dtypes'],'Fit schema differs')
 w=np.load(folder/'fit_weights.npy',allow_pickle=False);lr=np.load(folder/'fit_raw_log_ratio.npy',allow_pickle=False);groups,_=domain.group_keys(X);expected,_,summary=domain.weight_summary(lr,groups)
 need(w.shape==(len(X),) and np.isfinite(w).all() and np.all(w>0) and abs(w.mean()-1)<1e-12,'Invalid prepared weights')
 need(np.array_equal(w,expected) and summary==m['weight_summary'] and domain.value_sha(w)==m['fit_weight_values_sha256'],'Weight replay differs')
 contract={'phase':stage,**{k:m[k] for k in ['fit_X_sha256','query_X_sha256','fit_indices_content_sha256','query_indices_content_sha256']},'weights_values_sha256':m['fit_weight_values_sha256'],'domain_plan_sha256':m['domain_plan_sha256'],'preparation_manifest_sha256':sha(folder/'manifest.json'),'query_X_used_for_weights':True,'query_labels_used_for_weights':False}
 return w,contract,m

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('action',choices=['plan','prepare','verify']);parser.add_argument('--stage',choices=[*SPLITS,'final'],default='development');args=parser.parse_args()
 if args.action=='plan':plan_create()
 elif args.action=='prepare':prepare(args.stage)
 else:
  authorize(args.stage);X,Xq,fi,qi=partition_X(args.stage);w,contract,m=load_prepared(args.stage,X,Xq,fi,qi);print(json.dumps({'verified':True,'stage':args.stage,'weight_rows':len(w),'contract':contract}),flush=True)
