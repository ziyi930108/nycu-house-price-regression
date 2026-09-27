"""R9: one fixed historical-POI raw CatBoost replacement, compared with v3.
Commands never modify R6, previous submissions, shared caches or the registry.
"""
from __future__ import annotations
import argparse,ast,hashlib,json,time,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import round6_common as c6
import round6_evaluate as e6
import round9_geo_poi as component
from round6_prepare_geo import load_geographic_partition
from round6_geo import frame_sha
from round2_neighborhood import predict_reference
from round3_evaluate import score,target_weights,runtime_versions
ROOT=c6.ROOT
OUT=ROOT/'outputs/round9'
STAGES=tuple(c6.SPLITS)
CANDIDATE='poi_geography_blend'
CONTROL='v3_control'
STAGE_YEAR={'development':2022,'confirmation':2023,'auxiliary':2021,'final':2024}
EXTRA_SOURCES=('src/round9_prepare_poi.py','src/round9_poi_extract.py','src/round9_poi_inventory.py','src/geo_osm_extract.py','src/round9_poi_extract_annual.py')
REQUIRED_DOCUMENTS=('reports/round9_validation_plan.md','reports/round9_model_design_review.md','reports/round9_poi_design_review.md')
REQUIRED_AUDITS=('outputs/round9/audit/projection_geometry_check.json','outputs/round9/audit/poi_inventory_2022.json','outputs/round9/audit/poi_preparation_self_check.json','outputs/round9/poi_index/2022/audit.json','outputs/round9/audit/prepared_independent_review.json')
SNAPSHOT_SHA='77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
V3_SHA='52d5976cc988411d336130e722143d4933e71ae0888edf3dcfb71f1d13b68cd3'
V2_SHA='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
THRESHOLDS=dict(e6.THRESHOLDS)
FORMULAS={CONTROL:'.75*(.5*old_raw_cat+.5*geo_raw_cat)+.25*old_unit_lgb; then v2 trend once',CANDIDATE:'.75*(.5*old_raw_cat+.5*poi_geo_raw_cat)+.25*old_unit_lgb; then v2 trend once'}
require,read,write,locked,locked_write,sha,rel=c6.require,c6.read,c6.write,c6.read_locked,c6.locked_write,c6.sha256,c6.rel

def now():return datetime.now(timezone.utc).isoformat()
def value_sha(a,dtype):return hashlib.sha256(np.asarray(a,dtype=dtype).tobytes()).hexdigest()
def sources():
    pending=[Path(__file__).resolve()]+[ROOT/name for name in EXTRA_SOURCES];found=set()
    while pending:
        path=pending.pop()
        if path in found:continue
        found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            names=[item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for name in names:
                local=ROOT/'src'/(name.split('.')[0]+'.py')
                if local.is_file() and local not in found:pending.append(local)
    return {rel(p):sha(p) for p in sorted(found)}

def stage_paths(stage):
    base=c6.baseline_folder(stage)
    geo=c6.OUT/'models'/stage/'geography_catboost'
    evaluation=c6.OUT/'evaluation'/stage
    paths=[base/n for n in ['catboost_raw.npy','lightgbm_l1_unit63.npy','reference_v2.pkl','fit_indices.npy','check_indices.npy']]
    paths += [geo/n for n in ['metadata.json','predictions.npy','fit_indices.npy','query_indices.npy']]
    paths += [evaluation/n for n in ['results.json','results.sha256','geography_blend.npy']]
    manifest=c6.OUT/'geography'/stage/'manifest.json'
    paths += [manifest,manifest.with_suffix('.sha256')]
    item=locked(manifest)
    paths += [ROOT/v['path'] for v in item['features'].values()]
    paths += [ROOT/p for p in item['index_files_sha256']]
    paths += [ROOT/p for p in item['source_sha256']]
    return sorted(set(paths))

def json_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False,separators=(',',':'),allow_nan=False).encode('utf-8')).hexdigest()


def checked_path(name):
    path=(ROOT/Path(name)).resolve()
    require(path.is_relative_to(ROOT.resolve()) and path.is_file(),'Missing/unsafe dependency: '+str(name))
    return path


def load_model(path):
    model=CatBoostRegressor()
    model.load_model(str(path))
    return model


def check_later_gate(stage,frozen,plan_sha):
    require(stage in ('confirmation','auxiliary'),'Only declared later backtests may be authorized')
    require(frozen['passed'] is True and frozen['later_stages_allowed'] is True and frozen['selected_candidate']==CANDIDATE,'Development rejected; later stages prohibited')
    require(frozen['plan_sha256']==plan_sha,'Development selection belongs to another plan')


def validate_poi_contract(manifest,specification,stage):
    require(stage in STAGE_YEAR and manifest['round']==9 and manifest['stage']==stage and manifest['status']=='complete','Wrong prepared POI identity')
    require(manifest['rules']==specification['poi_rules'] and json_sha(manifest['rules'])==specification['poi_rules_sha256'],'POI preparation rules changed')
    require(manifest['rules']['columns']==component.POI_FEATURES and manifest['rules']['poi_keys']==list(component.POI_KEYS) and manifest['rules']['dtypes']=='float32','POI schema differs from fixed helper')
    require(manifest['versions']==specification['poi_preparation_versions'],'POI preparation runtime changed')
    annual=specification['annual_geography'][stage]
    for key in ['snapshot_year','source_pbf_sha256','replication_timestamp','forecast_origin_utc','geography_manifest_sha256']:
        require(manifest[key]==annual[key],'Annual POI metadata differs: '+key)
    require(datetime.fromisoformat(manifest['replication_timestamp'].replace('Z','+00:00'))<datetime.fromisoformat(manifest['forecast_origin_utc'].replace('Z','+00:00')),'Map is not earlier than forecast origin')
    require(set(manifest['features'])=={'train','valid','test','full'},'Prepared partition inventory differs')
    require(manifest['geographic_values_learned_from_targets'] is False and manifest['raw_workbooks_or_target_cache_deserialized'] is False and manifest['prediction_files_read'] is False,'POI preparation read prices or predictions')
    require('src/round9_prepare_poi.py' in manifest['source_sha256'],'POI preparation source is missing')
    for name,digest in manifest['source_sha256'].items():
        require(specification['source_sha256'].get(name)==digest,'Prepared source was not frozen: '+name)


def verify_poi_manifest(stage,specification):
    folder=OUT/'poi_features'/stage;path=folder/'manifest.json';manifest=locked(path)
    validate_poi_contract(manifest,specification,stage)
    inputs={rel(path):sha(path),rel(path.with_suffix('.sha256')):sha(path.with_suffix('.sha256'))}
    inputs.update(manifest['dependency_sha256'])
    known=dict(specification['source_sha256']);known.update(specification['dependency_sha256'])
    known.update(specification['stage_dependency_sha256'].get(stage,specification['final_dependency_sha256']))
    for name,digest in manifest['dependency_sha256'].items():
        if name in known:require(known[name]==digest,'Prepared dependency differs from frozen input: '+name)
        if name.startswith('src/') and name.endswith('.py'):require(name in known,'Unfrozen preparation dependency: '+name)
    annual=specification['annual_geography'][stage]
    pbf=checked_path(annual['source_pbf_path'])
    require(manifest['dependency_sha256'].get(rel(pbf))==annual['source_pbf_sha256'],'Prepared PBF not bound as a dependency')
    old_path=c6.OUT/'geography'/stage/'manifest.json';old=locked(old_path)
    require(sha(old_path)==manifest['geography_manifest_sha256'],'Prepared original geography manifest differs')
    require(old['source_pbf_sha256']==manifest['source_pbf_sha256'] and old['replication_timestamp']==manifest['replication_timestamp'],'POI and R6 maps differ')
    for name,item in manifest['features'].items():
        feature_path=checked_path(item['path'])
        require(feature_path==folder/(name+'.pkl'),'Prepared POI frame path differs')
        require(item['rows']==old['features'][name]['rows'],'Prepared and original geography row counts differ')
        inputs[rel(feature_path)]=item['sha256']
    coverage=checked_path(manifest['coverage']['path'])
    require(coverage==folder/'coverage.json','Coverage outside prepared stage')
    inputs[rel(coverage)]=manifest['coverage']['sha256']
    catalog_folder=OUT/'poi_index'/str(STAGE_YEAR[stage]);catalog_path=catalog_folder/'audit.json';catalog=locked(catalog_path)
    require(sha(catalog_path)==manifest['poi_index_audit_sha256'],'Prepared catalog receipt differs')
    require(catalog['status']=='complete' and catalog['year']==STAGE_YEAR[stage],'Catalog year or status differs')
    require(catalog['rules']==specification['poi_extraction_rules'] and catalog['versions']==specification['poi_extraction_versions'],'Catalog rules or runtime changed')
    require(checked_path(catalog['source']['path'])==pbf and catalog['source']['sha256']==annual['source_pbf_sha256'] and pbf.stat().st_size==catalog['source']['bytes'],'Catalog PBF identity differs')
    require(catalog['source']['header']['replication_timestamp']==annual['replication_timestamp'],'Catalog PBF header differs')
    require(catalog['source_before_after_verified'] is True and catalog['source_header_after']['replication_timestamp']==annual['replication_timestamp'],'Catalog source changed during extraction')
    require(catalog['ml_data_or_predictions_or_scores_read'] is False and catalog['external_property_or_transaction_answers_used'] is False,'Catalog used forbidden information')
    require({'src/round9_poi_extract.py','src/round9_poi_inventory.py','src/geo_osm_extract.py'}<=set(catalog['source_sha256']),'Catalog core sources missing')
    for name,digest in catalog['source_sha256'].items():
        require(specification['source_sha256'].get(name)==digest,'Unfrozen catalog source: '+name)
        inputs[name]=digest
    inputs[rel(catalog_path)]=sha(catalog_path);inputs[rel(catalog_path.with_suffix('.sha256'))]=sha(catalog_path.with_suffix('.sha256'))
    require({'pois','errors'}<=set(catalog['files']),'Incomplete extraction file catalog')
    for name,item in catalog['files'].items():
        artifact=checked_path(item['path']) if 'path' in item else catalog_folder/(name+'.jsonl')
        require(artifact.resolve().parent==catalog_folder.resolve() and artifact.is_file(),'Extraction artifact outside annual catalog')
        require(artifact.stat().st_size==item['bytes'],'Extraction artifact bytes differ')
        inputs[rel(artifact)]=item['sha256']
    for name,digest in catalog['input_evidence_sha256'].items():
        if name in known:require(known[name]==digest,'Catalog evidence differs from plan: '+name)
        inputs[name]=digest
    c6.verify_hashes(inputs)
    return manifest,inputs


def load_poi_partition(stage,fi,qi,geo,geoq,specification):
    require(stage in STAGES,'Only backtest POI partitions are loaded by this driver')
    manifest,inputs=verify_poi_manifest(stage,specification)
    item=manifest['features']['train'];train=pd.read_pickle(checked_path(item['path']))
    require(isinstance(train,pd.DataFrame) and list(train.columns)==component.POI_FEATURES and train.index.is_unique,'Prepared POI schema/index differs')
    require(str(train.index.dtype)=='int64' and len(train)==item['rows'],'Prepared POI row count/index dtype differs')
    require(frame_sha(train)==item['frame_sha256'] and value_sha(train.index,'<i8')==item['index_values_sha256'],'Prepared POI values/index hash differs')
    old=locked(c6.OUT/'geography'/stage/'manifest.json');original=pd.read_pickle(checked_path(old['features']['train']['path']))
    require(train.index.equals(original.index) and frame_sha(original)==item['geographic_frame_sha256'],'POI rows not aligned with declared geographic frame')
    component.check_poi(train,original)
    for label,indices in [('fit',fi),('query',qi)]:
        arr=np.asarray(indices);require(arr.ndim==1 and np.issubdtype(arr.dtype,np.integer) and ((arr>=0)&(arr<len(train))).all(),'Invalid POI '+label+' indices')
    poi=train.iloc[fi].reset_index(drop=True);poiq=train.iloc[qi].reset_index(drop=True)
    require(original.iloc[fi].reset_index(drop=True).equals(geo) and original.iloc[qi].reset_index(drop=True).equals(geoq),'Model geographic rows differ from POI preparation')
    component.check_poi(poi,geo);component.check_poi(poiq,geoq)
    return poi,poiq,manifest,inputs


def validate_model_contract(meta,specification,expected):
    require(meta['round']==9 and meta['component']==component.FEATURE_RULES['component'],'Wrong model component')
    require(meta['parameters']==specification['parameters'] and meta['feature_count']==109 and meta['categorical_feature_count']==23,'Model parameters/features differ')
    require(meta['iterations']==2999 and meta['actual_iterations']==2999 and meta['target_mode']=='raw' and meta['sample_weight']=='uniform','Model target or iteration contract differs')
    require(meta['target_formula']==component.FEATURE_RULES['target_formula'] and meta['prediction_formula']==component.FEATURE_RULES['prediction_formula'],'Model price scale differs')
    require(meta['target_used_for_geographic_state'] is False and meta['query_used_for_geographic_state'] is False and meta['reference_features_changed'] is False,'Feature construction used targets/query or changed reference')
    for name,value in expected.items():require(meta.get(name)==value,'Model metadata alignment differs: '+name)


def plan_create():
    require(not (OUT/'candidate_plan.json').exists(),'R9 plan exists; never overwrite')
    snapshot_path=ROOT/'runs/v3/snapshot_manifest.json'
    require(sha(snapshot_path)==SNAPSHOT_SHA,'Released v3 snapshot changed')
    snapshot=locked(snapshot_path)
    known={k:v['original_sha256'] for k,v in snapshot['copied_files'].items()}
    known.update({v['original_path']:v['sha256'] for v in snapshot['external_dependencies']})
    old=c6.plan();deps=dict(old['data_sha256'])
    required=[ROOT/'outputs/submission_v3.csv',ROOT/'outputs/submission_v2.csv',snapshot_path,snapshot_path.with_suffix('.sha256'),c6.OUT/'candidate_plan.json',c6.OUT/'candidate_plan.sha256',ROOT/'runs/v1/models/final_config.json',OUT/'audit/preflight.json']
    required += [ROOT/p for p in (*REQUIRED_DOCUMENTS,*REQUIRED_AUDITS)]
    for path in required:deps[rel(path)]=sha(path)
    require(deps['outputs/submission_v3.csv']==V3_SHA and deps['outputs/submission_v2.csv']==V2_SHA,'Preserved submissions changed')
    preflight=read(OUT/'audit/preflight.json')
    require(preflight['all_checks_passed'] is True and preflight['real_model_training_or_scoring'] is False,'Preflight did not pass')
    require(preflight['source_sha256']==sources(),'Source changed after preflight')
    stages={}
    for stage in STAGES:
        mapping={}
        for path in stage_paths(stage):
            name=rel(path);digest=sha(path)
            require(known.get(name)==digest,'Stage dependency differs from v3 snapshot: '+name)
            mapping[name]=digest
        stages[stage]=mapping
    final_manifest=c6.OUT/'geography/final/manifest.json';final_geo=locked(final_manifest)
    final_paths=[final_manifest,final_manifest.with_suffix('.sha256'),c6.OUT/'release/release.json',c6.OUT/'release/release.sha256']
    final_paths += [ROOT/v['path'] for v in final_geo['features'].values()]
    final_paths += [ROOT/p for field in ['source_sha256','index_files_sha256'] for p in final_geo[field]]
    final_dependencies={}
    for path in sorted(set(final_paths)):
        name=rel(path);digest=sha(path)
        require(known.get(name)==digest,'Final dependency differs from v3 snapshot: '+name)
        final_dependencies[name]=digest
    annual={}
    for stage,year in STAGE_YEAR.items():
        old_path=c6.OUT/'geography'/stage/'manifest.json';geo=locked(old_path)
        pbf=ROOT/'data_external/geography'/('taiwan-'+str(year)[2:]+'0101.osm.pbf')
        require(known.get(rel(pbf))==sha(pbf)==geo['source_pbf_sha256'],'Annual PBF differs from released source')
        require(geo['snapshot_year']==year,'Annual geographic year differs')
        deps[rel(pbf)]=sha(pbf)
        annual[stage]={'snapshot_year':year,'source_pbf_path':rel(pbf),'source_pbf_sha256':sha(pbf),'replication_timestamp':geo['replication_timestamp'],'forecast_origin_utc':geo['forecast_origin_utc'],'geography_manifest_sha256':sha(old_path)}
    dev=locked(OUT/'poi_features/development/manifest.json')
    catalog=locked(OUT/'poi_index/2022/audit.json')
    value={'round':9,'created_at_utc':now(),'candidate':CANDIDATE,'control':CONTROL,'formulas':FORMULAS,'thresholds':THRESHOLDS,'parameters':component.parameters(),'feature_count':109,'categorical_count':23,'feature_rules':component.FEATURE_RULES,'poi_rules':dev['rules'],'poi_rules_sha256':json_sha(dev['rules']),'poi_preparation_versions':dev['versions'],'poi_extraction_rules':catalog['rules'],'poi_extraction_versions':catalog['versions'],'annual_geography':annual,'source_sha256':sources(),'dependency_sha256':deps,'stage_dependency_sha256':stages,'final_dependency_sha256':final_dependencies,'final_contract':old['final_contract'],'splits':old['splits'],'split_contract':old['split_contract'],'target_group_weights':old['target_group_weights'],'quarter_definition':old['quarter_definition'],'old_plan_sha256':sha(c6.OUT/'candidate_plan.json'),'v3_snapshot_sha256':SNAPSHOT_SHA,'single_development_candidate':True,'tuning_or_weight_search':False,'public_score_used_for_selection':False,'historically_untouched':False,'external_property_answers_used':False,'fit_uses_eval_set':False,'early_stopping_used':False,'runtime':runtime_versions(),'registry_sha256_at_plan':sha(ROOT/'runs/versions.json'),'final_fit_implemented_in_this_driver':False,'later_poi_preparation_may_follow_development':True}
    _,prepared_inputs=verify_poi_manifest('development',value)
    deps.update(prepared_inputs)
    c6.verify_hashes(deps)
    require(value['source_sha256']==preflight['source_sha256']==sources(),'Sources changed after preflight or while freezing plan')
    locked_write(OUT/'candidate_plan.json',value)
    print(json.dumps({'status':'plan_frozen','plan_sha256':sha(OUT/'candidate_plan.json'),'candidate':CANDIDATE}),flush=True)


def plan():
    value=locked(OUT/'candidate_plan.json')
    require(value['round']==9 and value['candidate']==CANDIDATE and value['control']==CONTROL,'Wrong R9 plan')
    require(value['formulas']==FORMULAS and value['thresholds']==THRESHOLDS and value['parameters']==component.parameters(),'Fixed configuration changed')
    require(value['feature_rules']==component.FEATURE_RULES and value['feature_count']==109 and value['categorical_count']==23,'Feature contract changed')
    require(value['source_sha256']==sources(),'Frozen source closure changed')
    require(value['poi_rules_sha256']==json_sha(value['poi_rules']),'Frozen POI rules changed')
    require(value['splits']=={k:list(v) for k,v in c6.SPLITS.items()},'Split definitions changed')
    require(value['runtime']==runtime_versions(),'Runtime versions changed')
    c6.verify_hashes(value['dependency_sha256'])
    require(sha(ROOT/'runs/versions.json')==value['registry_sha256_at_plan'],'Preserved registry changed')
    return value


def authorize(stage):
    require(stage!='final','Final fitting is not implemented by this frozen backtest driver; accepted three-stage results are required before any separate final/release runner')
    require(stage in STAGES,'Only fixed backtest stages allowed')
    value=plan();c6.verify_hashes(value['stage_dependency_sha256'][stage])
    if stage!='development':
        frozen=locked(OUT/'frozen_candidate.json')
        check_later_gate(stage,frozen,sha(OUT/'candidate_plan.json'))
        require(frozen['source_sha256']==value['source_sha256'] and frozen['parameters']==value['parameters'] and frozen['formulas']==FORMULAS and frozen['thresholds']==THRESHOLDS and frozen['poi_rules_sha256']==value['poi_rules_sha256'],'Development freeze protocol changed')
        c6.verify_hashes(frozen['development_artifact_sha256'])
        require(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'Development result differs')
        development=read_result('development')
        require(development['decision']['passed'] is True and development['decision']['selected_candidate']==CANDIDATE,'Development gates did not pass')
    return value


def partition(stage):
    data,X,y,Xq,yq,fi,qi=c6.partition(stage)
    spec=locked(OUT/'candidate_plan.json')['split_contract'][stage]
    for label,array in [('fit',fi),('query',qi)]:
        require(spec[label+'_indices_content_sha256']==value_sha(array,'<i8'),'R9 partition index differs')
    base=c6.baseline_folder(stage);old=c6.OUT/'models'/stage/'geography_catboost'
    e6.verify_indices(base,fi,qi,'check_indices.npy');e6.verify_indices(old,fi,qi,'query_indices.npy')
    return data,X,y,Xq,yq,fi,qi

def publish(staging,destination):
    require(staging.resolve().parent==destination.resolve().parent and staging.resolve().is_relative_to(OUT.resolve()),'Unsafe R9 publication path')
    require(not destination.exists(),'Refusing to replace recorded experiment')
    staging.rename(destination)

def fit(stage):
    specification=authorize(stage);destination=OUT/'models'/stage
    require(not destination.exists(),'Model already exists; do not retrain')
    data,X,y,Xq,yq,fi,qi=partition(stage);del yq
    geo,geoq,manifest=load_geographic_partition(stage,fi,qi)
    poi,poiq,poi_manifest,poi_inputs=load_poi_partition(stage,fi,qi,geo,geoq,specification)
    destination.parent.mkdir(parents=True,exist_ok=True)
    staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    started=time.monotonic();cats=data['categorical_columns']
    print(json.dumps({'event':'fit_started','stage':stage,'fit_rows':len(X),'query_rows':len(Xq),'features':109,'parameters':specification['parameters']}),flush=True)
    model,state,metadata=component.fit_model(X,y,geo,poi,cats)
    predictions=component.predict_model(model,state,Xq,geoq,poiq,cats)
    model.save_model(str(staging/'model.cbm'));write(staging/'feature_state.json',state)
    reloaded=load_model(staging/'model.cbm')
    replay=component.predict_model(reloaded,read(staging/'feature_state.json'),Xq.iloc[:64],geoq.iloc[:64],poiq.iloc[:64],cats)
    require(np.allclose(replay,predictions[:64],rtol=1e-12,atol=1e-5),'Saved model reload differs')
    require(int(reloaded.tree_count_)==2999,'Unexpected CatBoost iteration count')
    require(predictions.shape==(len(Xq),) and np.isfinite(predictions).all() and (predictions>=0).all(),'Invalid TWD predictions')
    for name,values in [('fit_indices',fi),('query_indices',qi),('predictions',predictions)]:np.save(staging/(name+'.npy'),values,allow_pickle=False)
    metadata.update({'round':9,'stage':stage,'component':component.FEATURE_RULES['component'],'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'source_sha256':specification['source_sha256'],'outer_fit_rows':len(X),'query_rows':len(Xq),'fit_target_sha256':value_sha(y,'<f8'),'fit_X_sha256':frame_sha(X),'query_X_sha256':frame_sha(Xq),'fit_geo_sha256':frame_sha(geo),'query_geo_sha256':frame_sha(geoq),'fit_poi_sha256':frame_sha(poi),'query_poi_sha256':frame_sha(poiq),'poi_manifest_sha256':sha(OUT/'poi_features'/stage/'manifest.json'),'poi_rules_sha256':specification['poi_rules_sha256'],'geo_manifest_sha256':sha(c6.OUT/'geography'/stage/'manifest.json'),'artifact_sha256':{p.name:sha(p) for p in staging.iterdir() if p.is_file()},'reload_64_rows_verified':True,'query_labels_used':False,'prediction_units':'TWD','direct_component_score_omitted':True,'runner_elapsed_seconds':time.monotonic()-started,'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json')})
    locked_write(staging/'metadata.json',metadata)
    authorize(stage);verify_poi_manifest(stage,specification);publish(staging,destination)
    print(json.dumps({'event':'fit_completed','stage':stage,'elapsed_seconds':time.monotonic()-started}),flush=True)

def blend(old,geo,unit,reference,Xq,apply_reference=predict_reference):
    base=.75*(.5*np.asarray(old)+.5*np.asarray(geo))+.25*np.asarray(unit)
    prediction=np.asarray(apply_reference(reference,Xq,base,variant='trend'),dtype='float64')
    require(prediction.shape==(len(Xq),) and np.isfinite(prediction).all() and (prediction>=0).all(),'Invalid blended TWD predictions')
    return prediction

def decide(stage,metrics):
    require(set(metrics)=={CONTROL,CANDIDATE},'Only the fixed control and candidate may be scored')
    base,value=metrics[CONTROL],metrics[CANDIDATE]
    large=e6.large_strata_ok(value,base)
    improved=[q for q,item in base['forecast_quarters'].items() if item['rows'] and value['forecast_quarters'][q]['mae']<item['mae']]
    if stage=='development':
        checks={'overall_improves_at_least_1pct':value['overall_mae']<=(1-THRESHOLDS['development_min_overall_improvement'])*base['overall_mae'],'weighted_improves_at_least_1pct':value['weighted_mae']<=(1-THRESHOLDS['development_min_weighted_improvement'])*base['weighted_mae'],'quarters_deteriorate_at_most_5pct':all(value['forecast_quarters'][q]['mae']<=(1+THRESHOLDS['development_max_quarter_deterioration'])*item['mae'] for q,item in base['forecast_quarters'].items() if item['rows']),'large_strata_deteriorate_at_most_5pct':large}
    else:
        require(stage in ('confirmation','auxiliary'),'Unknown evaluation stage')
        checks={'overall_within_limit':value['overall_mae']<=(1+THRESHOLDS[stage+'_max_overall_deterioration'])*base['overall_mae'],'weighted_within_limit':value['weighted_mae']<=(1+THRESHOLDS[stage+'_max_weighted_deterioration'])*base['weighted_mae'],'large_strata_deteriorate_at_most_5pct':large}
        if stage=='confirmation':checks['at_least_two_quarters_improve']=len(improved)>=THRESHOLDS['confirmation_min_improved_quarters']
    passed=all(checks.values())
    return {'passed':passed,'selected_candidate':CANDIDATE if passed else CONTROL,'checks':checks,'improved_quarters':improved,'candidate_reselection_performed':False}

def model_check(stage,X,y,Xq,fi,qi,geo,geoq,poi,poiq,cats,specification):
    folder=OUT/'models'/stage;meta=locked(folder/'metadata.json')
    expected={key:frame_sha(frame) for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq),('fit_poi_sha256',poi),('query_poi_sha256',poiq)]}
    expected.update({'poi_manifest_sha256':sha(OUT/'poi_features'/stage/'manifest.json'),'poi_rules_sha256':specification['poi_rules_sha256']})
    validate_model_contract(meta,specification,expected)
    require(meta['round']==9 and meta['stage']==stage and meta['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong model identity')
    require(meta['source_sha256']==specification['source_sha256'],'Model source closure differs')
    require(meta['parameters']==specification['parameters'] and meta['feature_count']==109 and meta['categorical_feature_count']==23,'Model configuration differs')
    require(meta['fit_uses_eval_set'] is False and meta['early_stopping_used'] is False and meta['sample_weight']=='uniform','Fit contract differs')
    require(meta['fit_transformed_target_values_sha256']==value_sha(np.asarray(y,dtype='float64')/1e6,'<f8'),'Transformed fit target differs')
    require(meta['outer_fit_rows']==len(X) and meta['query_rows']==len(Xq),'Model sample counts differ')
    require(meta['fit_target_sha256']==value_sha(y,'<f8'),'Fit target identity differs')
    for key,frame in [('fit_X_sha256',X),('query_X_sha256',Xq),('fit_geo_sha256',geo),('query_geo_sha256',geoq),('fit_poi_sha256',poi),('query_poi_sha256',poiq)]:require(meta[key]==frame_sha(frame),'Model feature identity differs: '+key)
    require(meta['geo_manifest_sha256']==sha(c6.OUT/'geography'/stage/'manifest.json'),'Model map version differs')
    require(meta['reload_64_rows_verified'] is True and meta['query_labels_used'] is False and meta['prediction_units']=='TWD','Model training/prediction contract differs')
    expected_freeze=None if stage=='development' else sha(OUT/'frozen_candidate.json')
    require(meta['frozen_candidate_sha256']==expected_freeze,'Model selection identity differs')
    for name,digest in meta['artifact_sha256'].items():
        require(Path(name).name==name and sha(folder/name)==digest,'Saved model artifact changed')
    e6.verify_indices(folder,fi,qi,'query_indices.npy')
    saved=e6.vector(folder/'predictions.npy',len(qi))
    model=load_model(folder/'model.cbm')
    require(int(model.tree_count_)==2999,'Reloaded model iteration count differs')
    state=read(folder/'feature_state.json')
    require(component.fit_feature_state(X,geo,poi,cats)==state,'Saved feature state differs from exact fit-only state')
    replay=component.predict_model(model,state,Xq,geoq,poiq,cats)
    require(np.allclose(saved,replay,rtol=1e-12,atol=1e-5),'All-query model replay differs')
    return saved,{rel(p):sha(p) for p in folder.iterdir() if p.is_file()}

def read_result(stage):
    path=OUT/'evaluation'/stage/'results.json';value=locked(path)
    require(value['stage']==stage and value['plan_sha256']==sha(OUT/'candidate_plan.json'),'Wrong saved result')
    require(value['source_sha256']==sources() and value['formulas']==FORMULAS and value['thresholds']==THRESHOLDS,'Saved result protocol differs')
    require(value['frozen_candidate_sha256']==(None if stage=='development' else sha(OUT/'frozen_candidate.json')),'Saved result freeze differs')
    require(value['decision']==decide(stage,value['metrics']),'Saved gate decision differs')
    for field in ['input_artifact_sha256','output_artifact_sha256']:c6.verify_hashes(value[field])
    return value

def freeze_development(result):
    path=OUT/'frozen_candidate.json'
    if path.exists():
        frozen=locked(path)
        require(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'Existing freeze result differs')
        return
    artifacts={rel(p):sha(p) for folder in [OUT/'models/development',OUT/'evaluation/development'] for p in folder.iterdir() if p.is_file()}
    value={'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'result_sha256':sha(OUT/'evaluation/development/results.json'),'passed':result['decision']['passed'],'selected_candidate':result['decision']['selected_candidate'],'development_artifact_sha256':artifacts,'source_sha256':sources(),'formulas':FORMULAS,'thresholds':THRESHOLDS,'parameters':component.parameters(),'poi_rules_sha256':plan()['poi_rules_sha256'],'later_stages_allowed':result['decision']['passed'],'historically_untouched':False}
    locked_write(path,value)

def evaluate(stage):
    specification=authorize(stage);destination=OUT/'evaluation'/stage
    if destination.exists():
        result=read_result(stage)
        if stage=='development':freeze_development(result)
        print(json.dumps({'status':'existing_result_verified','stage':stage,'decision':result['decision']}),flush=True)
        return result
    data,X,y,Xq,yq,fi,qi=partition(stage)
    weights=target_weights(data)
    require(weights==specification['target_group_weights'],'Fixed test-X grouping weights differ')
    geo,geoq,_=load_geographic_partition(stage,fi,qi)
    poi,poiq,poi_manifest,poi_inputs=load_poi_partition(stage,fi,qi,geo,geoq,specification)
    new,inputs=model_check(stage,X,y,Xq,fi,qi,geo,geoq,poi,poiq,data['categorical_columns'],specification)
    inputs.update(poi_inputs)
    inputs.update(specification['stage_dependency_sha256'][stage])
    reference,_=c6.load_reference(stage,X);base=c6.baseline_folder(stage)
    old=e6.vector(base/'catboost_raw.npy',len(qi));unit=e6.vector(base/'lightgbm_l1_unit63.npy',len(qi))
    geographic=e6.vector(c6.OUT/'models'/stage/'geography_catboost/predictions.npy',len(qi))
    original=e6.vector(c6.OUT/'evaluation'/stage/'geography_blend.npy',len(qi))
    replay=blend(old,geographic,unit,reference,Xq)
    require(np.allclose(original,replay,rtol=1e-12,atol=1e-5),'v3 control does not replay')
    predictions={CONTROL:original,CANDIDATE:blend(old,new,unit,reference,Xq)}
    metrics={name:score(yq,pred,X,Xq,c6.SPLITS[stage][0],weights) for name,pred in predictions.items()}
    previous=locked(c6.OUT/'evaluation'/stage/'results.json')
    require(metrics[CONTROL]==previous['metrics']['geography_blend'],'Recomputed v3 metrics differ from archived result')
    decision=decide(stage,metrics)
    destination.parent.mkdir(parents=True,exist_ok=True);staging=destination.parent/('.'+stage+'-'+uuid.uuid4().hex);staging.mkdir()
    for name,values in predictions.items():np.save(staging/(name+'.npy'),values,allow_pickle=False)
    outputs={rel(destination/p.name):sha(p) for p in staging.iterdir() if p.is_file()}
    result={'round':9,'stage':stage,'created_at_utc':now(),'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':None if stage=='development' else sha(OUT/'frozen_candidate.json'),'source_sha256':specification['source_sha256'],'input_artifact_sha256':inputs,'output_artifact_sha256':outputs,'formulas':FORMULAS,'thresholds':THRESHOLDS,'target_group_weights':weights,'metrics':metrics,'decision':decision,'v3_control_replayed_all_rows':True,'new_model_replayed_all_rows':True,'fit_rows':len(fi),'query_rows':len(qi),'evaluated_candidates':[CONTROL,CANDIDATE],'candidate_component_separately_scored':False,'query_labels_used_for_training':False,'public_score_used':False,'historically_untouched':False,'new_public_score_known':False,'private_score_known':False}
    locked_write(staging/'results.json',result)
    authorize(stage);verify_poi_manifest(stage,specification);publish(staging,destination)
    if stage=='development':freeze_development(result)
    print(json.dumps({'stage':stage,'metrics':{k:{n:v[n] for n in ['overall_mae','weighted_mae']} for k,v in metrics.items()},'decision':decision}),flush=True)
    return result

def finalize():
    specification=plan();frozen=locked(OUT/'frozen_candidate.json');development=read_result('development')
    require(frozen['plan_sha256']==sha(OUT/'candidate_plan.json') and frozen['source_sha256']==specification['source_sha256'] and frozen['poi_rules_sha256']==specification['poi_rules_sha256'],'Finalization freeze identity differs')
    require(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json') and frozen['passed']==development['decision']['passed'],'Development freeze differs')
    results={'development':development}
    if frozen['passed']:
        for stage in ('confirmation','auxiliary'):
            authorize(stage);results[stage]=read_result(stage)
    if not frozen['passed']:
        for stage in ('confirmation','auxiliary'):
            require(not (OUT/'models'/stage).exists() and not (OUT/'evaluation'/stage).exists(),'Rejected development has unauthorized later model/results')
    accepted=all(r['decision']['passed'] for r in results.values()) and len(results)==3
    path=OUT/'accepted_candidate.json'
    result={'created_at_utc':now(),'round':9,'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),'accepted':accepted,'candidate':CANDIDATE,'deployment_decision':CANDIDATE if accepted else 'keep_v3','result_sha256':{s:sha(OUT/'evaluation'/s/'results.json') for s in results},'unevaluated_stages':[s for s in STAGES if s not in results],'no_reselection_or_parameter_change':True,'new_submission_created':False,'private_score_known':False,'source_sha256':specification['source_sha256']}
    if path.exists():
        saved=locked(path);require(all(saved[k]==v for k,v in result.items() if k!='created_at_utc'),'Existing final decision differs');result=saved
    else:locked_write(path,result)
    print(json.dumps(result),flush=True)
    return result

def self_check(write_output=True):
    import copy
    base={'overall_mae':100.,'weighted_mae':100.,'strata':{'seen':{'rows':500,'mae':100.},'missing':{'rows':0,'mae':None}},'forecast_quarters':{q:{'rows':10,'mae':100.} for q in ['Q1','Q2','Q3','Q4']}}
    checks=[]
    component_checks=component.self_check()
    def run(label,truth):require(truth,label);checks.append(label)
    def metrics(overall,weighted=None):
        value=copy.deepcopy(base);value['overall_mae']=overall;value['weighted_mae']=overall if weighted is None else weighted
        return {CONTROL:copy.deepcopy(base),CANDIDATE:value}
    run('development_exact_1pct_passes',decide('development',metrics(99))['passed'])
    run('development_below_1pct_rejected',not decide('development',metrics(99.00001))['passed'])
    run('weighted_also_required',not decide('development',metrics(98,100))['passed'])
    m=metrics(98);m[CANDIDATE]['strata']['seen']['mae']=105.00001
    run('500_row_stratum_gate',not decide('development',m)['passed'])
    m=metrics(98);m[CANDIDATE]['forecast_quarters']['Q1']['mae']=105.00001
    run('quarter_gate',not decide('development',m)['passed'])
    m=metrics(99);m[CANDIDATE]['forecast_quarters']['Q1']['mae']=99
    run('confirmation_one_quarter_insufficient',not decide('confirmation',m)['passed'])
    m[CANDIDATE]['forecast_quarters']['Q2']['mae']=99
    run('confirmation_two_quarters_pass',decide('confirmation',m)['passed'])
    run('auxiliary_exact_3pct_passes',decide('auxiliary',metrics(103))['passed'])
    run('auxiliary_over_3pct_rejected',not decide('auxiliary',metrics(103.00001))['passed'])
    calls=[];query=[object(),object()]
    def fake(reference,X,values,variant):
        require(X is query and reference=='fit_only' and variant=='trend','Reference call changed');calls.append(values.copy());return values*1.1
    result=blend([1,2],[3,4],[5,6],'fit_only',query,fake)
    run('fixed_weights_and_single_trend',len(calls)==1 and np.allclose(result,1.1*(.375*np.array([1,2])+.375*np.array([3,4])+.25*np.array([5,6]))))
    extras=additional_self_checks()
    checks.extend(extras)
    proof={'created_at_utc':now(),'all_checks_passed':True,'synthetic_checks':checks,'component_synthetic_checks':component_checks,'source_sha256':sources(),'real_model_training_or_scoring':False,'parameters':component.parameters()}
    if write_output:
        path=OUT/'audit/preflight.json';require(not path.exists(),'Preflight exists; preserve audit')
        write(path,proof)
    print(json.dumps(proof),flush=True)
    return proof


def additional_self_checks():
    import copy
    checks=[]
    def rejected(name,operation):
        try:operation()
        except (RuntimeError,ValueError,AssertionError,KeyError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    selected={'passed':True,'later_stages_allowed':True,'selected_candidate':CANDIDATE,'plan_sha256':'fixed'}
    check_later_gate('confirmation',selected,'fixed');check_later_gate('auxiliary',selected,'fixed')
    checks.append('both_later_stages_require_same_passed_frozen_candidate')
    for key,value in [('passed',False),('later_stages_allowed',False),('selected_candidate',CONTROL),('plan_sha256','different')]:
        bad=dict(selected);bad[key]=value
        rejected('later_gate_rejects_'+key,lambda:check_later_gate('confirmation',bad,'fixed'))
    rejected('final_fit_not_implemented_in_backtest_driver',lambda:check_later_gate('final',selected,'fixed'))
    rules={'columns':component.POI_FEATURES,'poi_keys':list(component.POI_KEYS),'dtypes':'float32'}
    annual={'snapshot_year':2022,'source_pbf_sha256':'pbf','replication_timestamp':'2022-01-01T00:00:00Z','forecast_origin_utc':'2022-08-10T16:00:00Z','geography_manifest_sha256':'geo'}
    specification={'poi_rules':rules,'poi_rules_sha256':json_sha(rules),'poi_preparation_versions':{'python':'fixed'},'annual_geography':{'development':annual},'source_sha256':{'src/round9_prepare_poi.py':'source'},'parameters':component.parameters()}
    manifest={'round':9,'stage':'development','status':'complete','rules':rules,'versions':{'python':'fixed'},**annual,'features':{k:{} for k in ['train','valid','test','full']},'geographic_values_learned_from_targets':False,'raw_workbooks_or_target_cache_deserialized':False,'prediction_files_read':False,'source_sha256':{'src/round9_prepare_poi.py':'source'}}
    validate_poi_contract(manifest,specification,'development');checks.append('fixed_annual_poi_manifest_contract')
    for key,value in [('snapshot_year',2023),('source_pbf_sha256','wrong'),('geography_manifest_sha256','wrong'),('replication_timestamp','2024-01-01T00:00:00Z'),('prediction_files_read',True),('source_sha256',{'src/round9_prepare_poi.py':'changed'})]:
        bad=copy.deepcopy(manifest);bad[key]=value
        rejected('poi_manifest_rejects_'+key,lambda:validate_poi_contract(bad,specification,'development'))
    model={'round':9,'component':component.FEATURE_RULES['component'],'parameters':component.parameters(),'feature_count':109,'categorical_feature_count':23,'iterations':2999,'actual_iterations':2999,'target_mode':'raw','sample_weight':'uniform','target_formula':component.FEATURE_RULES['target_formula'],'prediction_formula':component.FEATURE_RULES['prediction_formula'],'target_used_for_geographic_state':False,'query_used_for_geographic_state':False,'reference_features_changed':False,'fit_poi_sha256':'fit','query_poi_sha256':'query','poi_manifest_sha256':'manifest'}
    expected={'fit_poi_sha256':'fit','query_poi_sha256':'query','poi_manifest_sha256':'manifest'}
    validate_model_contract(model,specification,expected);checks.append('model_metadata_alignment_contract')
    for key in expected:
        bad=copy.deepcopy(model);bad[key]='wrong'
        rejected('model_metadata_rejects_'+key,lambda:validate_model_contract(bad,specification,expected))
    bad=copy.deepcopy(model);bad['sample_weight']='area/100'
    rejected('unit_model_cannot_pass_raw_model_contract',lambda:validate_model_contract(bad,specification,expected))
    return checks

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['self-check','plan','fit','evaluate','finalize'])
    parser.add_argument('--stage',choices=(*STAGES,'final'),default='development')
    parser.add_argument('--no-write',action='store_true',help='Synthetic checks only; do not publish preflight')
    args=parser.parse_args()
    if args.command=='self-check':self_check(write_output=not args.no_write)
    elif args.command=='plan':plan_create()
    elif args.command=='fit':fit(args.stage)
    elif args.command=='evaluate':evaluate(args.stage)
    else:finalize()
