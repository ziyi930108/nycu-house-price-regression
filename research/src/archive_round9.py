"""Archive finalized R9 backtests without fitting, scoring, release or registry changes.

Default creates runs/round9_experiments once; --verify is read-only.
--self-check exercises synthetic contracts only and writes no artifacts.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import archive_round7 as prior
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round9';DEST=ROOT/'runs/round9_experiments'
STAGES=('development','confirmation','auxiliary')
YEARS={'development':2022,'confirmation':2023,'auxiliary':2021,'final':2024}
CONTROL='v3_control';CANDIDATE='poi_geography_blend'
MANIFEST='snapshot_manifest.json';RECEIPT='snapshot_manifest.sha256';NOTICE_FILE='SNAPSHOT_README.md'
EXTRA_SOURCES=('round9_prepare_poi.py','round9_poi_extract.py','round9_poi_inventory.py','geo_osm_extract.py','round9_poi_extract_annual.py')
REPORTS=('README_round9.md','reports/round9_report.md','reports/round9_results.json','reports/round9_work_log.md','reports/round9_validation_plan.md','reports/round9_poi_design_review.md','reports/round9_model_design_review.md')
PREVIOUS={7:'48184abad594cccf216b6ba154d6704738d35fd36ddfb58077423c931e2a7b60',8:'8f352cc3957cc5b1dacaecb350e50fdea406350b2433dae1fd17052eb5b0493a'}
NOTICE='''# R9 finalized backtest snapshot

This preserves the fixed R9 models, predictions, stage decisions, receipts,
sources, all four annual POI catalogs and prepared features, audits and reports.
It does not publish a model release, create a submission, upload to Kaggle or
modify the versions registry. An accepted result means only that the fixed
historical backtest gates passed; final fitting and release are still required.
The released recommendation remains outputs/submission_v3.csv.

Raw PBF maps, supplied spreadsheets, old feature caches, prior models, sources
and snapshots remain SHA-bound external dependencies at their original paths.
This snapshot is not self-contained. The old annual geographic manifests and
their complete source/index dependencies are recorded. No new Public or Private
score or wholly untouched historical test is claimed. Failed staging is kept.
The manifest excludes itself and has a separate SHA256 receipt.
'''
need,sha,read,rel,safe,encoded=prior.need,prior.sha,prior.read,prior.rel,prior.safe,prior.encoded

def completed_stages(frozen):
    need(type(frozen['passed']) is bool and frozen['later_stages_allowed'] is frozen['passed'],'Invalid frozen authorization')
    need(frozen['selected_candidate']==(CANDIDATE if frozen['passed'] else CONTROL),'Invalid frozen selection')
    return list(STAGES) if frozen['passed'] else ['development']

def validate_decision(stage,d):
    keys={'large_strata_deteriorate_at_most_5pct'}
    if stage=='development':keys|={'overall_improves_at_least_1pct','weighted_improves_at_least_1pct','quarters_deteriorate_at_most_5pct'}
    else:
        need(stage in STAGES,'Unknown decision stage');keys|={'overall_within_limit','weighted_within_limit'}
        if stage=='confirmation':keys.add('at_least_two_quarters_improve')
    need(set(d['checks'])==keys and all(type(v) is bool for v in d['checks'].values()),'Wrong gate schema')
    need(type(d['passed']) is bool and d['passed']==all(d['checks'].values()),'Gate flags inconsistent')
    need(d['selected_candidate']==(CANDIDATE if d['passed'] else CONTROL) and d['candidate_reselection_performed'] is False,'Unexpected selection')
    need(isinstance(d['improved_quarters'],list) and len(set(d['improved_quarters']))==len(d['improved_quarters']) and set(d['improved_quarters'])<={'Q1','Q2','Q3','Q4'},'Wrong quarter identity')

def copy_allowed(name):
    path=safe(ROOT,name);name=rel(path)
    allowed=name.startswith('outputs/round9/') or (name.startswith('src/') and (path.name.startswith('round9') or path.name=='archive_round9.py')) or (name.startswith('reports/') and 'round9' in path.name) or name=='README_round9.md'
    need(allowed and path.suffix.lower()!='.pbf','File outside R9 copy scope: '+name)
    return path

def verify_prior_experiments(add,locked):
    summary={}
    for round_,digest in PREVIOUS.items():
        folder=ROOT/'runs'/('round%d_experiments'%round_);path=folder/MANIFEST
        add(path,digest,'preserved experiment snapshot');value=locked(path,'preserved experiment receipt')
        need(value['experiment_round']==round_ and value['is_release'] is False,'Wrong prior experiment')
        need(value['copied_files'].keys()==value['original_plan']['originals'].keys(),'Prior snapshot copy inventory differs')
        expected=set(value['copied_files'])|{MANIFEST,RECEIPT,value['dependency_notice']}
        need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==expected,'Prior snapshot inventory differs')
        for name,item in value['copied_files'].items():
            original=value['original_plan']['originals'][name]
            need(item['original_path']==name and item['backup_path']==rel(folder/name),'Prior snapshot path differs')
            need(item['original_sha256']==item['backup_sha256']==original['sha256'] and item['original_bytes']==item['backup_bytes']==original['bytes'],'Prior snapshot copy proof differs')
            for p in [safe(ROOT,name),safe(folder,name)]:
                need(p.stat().st_size==item['original_bytes'],'Prior snapshot file size differs');add(p,item['original_sha256'],'preserved prior experiment file')
        need(value['external_dependencies']==value['original_plan']['external_dependencies'],'Prior external inventory differs')
        for item in value['external_dependencies']:
            p=safe(ROOT,item['original_path']);need(p.stat().st_size==item['bytes'],'Prior external size differs');add(p,item['sha256'],'preserved prior experiment external')
        # Notices are literal constants in their archived source, not inferred prose.
        import ast
        tree=ast.parse((folder/'src'/('archive_round%d.py'%round_)).read_text(encoding='utf-8-sig'))
        notice=[ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='NOTICE' for t in n.targets)]
        need(len(notice)==1 and (folder/value['dependency_notice']).read_bytes()==notice[0].encode('utf-8'),'Prior notice differs')
        add(folder/value['dependency_notice'],reason='preserved prior notice')
        summary[str(round_)]={'manifest_sha256':digest,'copied_files':len(value['copied_files'])}
    return summary

def verify_poi(stage,plan,add,mapping,locked):
    year=YEARS[stage];annual=plan['annual_geography'][stage]
    old_path=ROOT/'outputs/round6/geography'/stage/'manifest.json';old=locked(old_path,'old geography provenance')
    need(sha(old_path)==annual['geography_manifest_sha256'] and old['geography_provenance_chain']['verified'] is True,'Old geography identity differs')
    for field in ['source_sha256','index_files_sha256']:mapping(old[field],stage+' full old geographic '+field)
    folder=OUT/'poi_index'/str(year);catalog=locked(folder/'audit.json',stage+' POI catalog')
    need(catalog['status']=='complete' and catalog['year']==year and catalog['rules']==plan['poi_extraction_rules'] and catalog['versions']==plan['poi_extraction_versions'],'POI catalog contract differs')
    source=catalog['source'];pbf=safe(ROOT,source['path'])
    need(source['path']==annual['source_pbf_path'] and source['sha256']==annual['source_pbf_sha256']==old['source_pbf_sha256'] and pbf.stat().st_size==source['bytes'],'POI PBF identity differs')
    need(source['header']['replication_timestamp']==catalog['source_header_after']['replication_timestamp']==annual['replication_timestamp']==old['replication_timestamp'],'POI PBF timestamp differs')
    need(catalog['source_before_after_verified'] is True and catalog['ml_data_or_predictions_or_scores_read'] is False and catalog['external_property_or_transaction_answers_used'] is False,'POI provenance policy differs')
    need(datetime.fromisoformat(annual['replication_timestamp'].replace('Z','+00:00'))<datetime.fromisoformat(annual['forecast_origin_utc'].replace('Z','+00:00')),'Map later than forecast origin')
    add(pbf,source['sha256'],'original annual PBF; external only')
    required={'src/round9_poi_extract.py','src/round9_poi_inventory.py','src/geo_osm_extract.py'}
    if year!=2022:required.add('src/round9_poi_extract_annual.py')
    need(required<=set(catalog['source_sha256']),'Missing extraction source')
    need(catalog['source_sha256'].get(catalog['script']['path'])==catalog['script']['sha256'],'Extraction entrypoint not bound')
    for name,digest in catalog['source_sha256'].items():need(plan['source_sha256'].get(name)==digest,'Unfrozen extraction source')
    mapping(catalog['source_sha256'],stage+' catalog sources');mapping(catalog['input_evidence_sha256'],stage+' catalog input evidence')
    need(len(catalog['synthetic_checks'])==14 and {'pois','errors'}<=set(catalog['files']),'Incomplete extraction catalog')
    for name,item in catalog['files'].items():
        p=safe(ROOT,item['path']);need(p.parent==folder.resolve() and p.stat().st_size==item['bytes'],'Catalog file path/size differs');add(p,item['sha256'],stage+' catalog file '+name)
    prepared=OUT/'poi_features'/stage;manifest=locked(prepared/'manifest.json',stage+' prepared POI')
    need(manifest['round']==9 and manifest['stage']==stage and manifest['status']=='complete','Wrong prepared identity')
    need(manifest['rules']==plan['poi_rules'] and manifest['versions']==plan['poi_preparation_versions'],'Prepared rules/runtime differ')
    for key in ['snapshot_year','source_pbf_sha256','replication_timestamp','forecast_origin_utc','geography_manifest_sha256']:need(manifest[key]==annual[key],'Prepared annual identity differs: '+key)
    need(manifest['poi_index_audit_sha256']==sha(folder/'audit.json'),'Prepared catalog differs')
    for key in ['geographic_values_learned_from_targets','raw_workbooks_or_target_cache_deserialized','prediction_files_read']:need(manifest[key] is False,'Prepared data policy differs')
    for name,digest in manifest['source_sha256'].items():need(plan['source_sha256'].get(name)==digest,'Unfrozen preparation source')
    mapping(manifest['source_sha256'],stage+' preparation sources');mapping(manifest['dependency_sha256'],stage+' preparation dependency')
    need(manifest['dependency_sha256'].get(source['path'])==source['sha256'],'Prepared PBF not bound')
    need(set(manifest['features'])=={'train','valid','test','full'},'Prepared partition inventory differs')
    for name,item in manifest['features'].items():
        p=safe(ROOT,item['path']);need(p==prepared/(name+'.pkl') and item['rows']==old['features'][name]['rows'],'Prepared path/rows differ');add(p,item['sha256'],stage+' prepared '+name)
    p=safe(ROOT,manifest['coverage']['path']);need(p==prepared/'coverage.json','Wrong coverage path');add(p,manifest['coverage']['sha256'],stage+' coverage')
    return manifest
def verify_model(stage,plan,fsha,add,locked):
    folder=OUT/'models'/stage;meta=locked(folder/'metadata.json',stage+' model metadata');spec=plan['split_contract'][stage]
    artifacts={'model.cbm','feature_state.json','predictions.npy','fit_indices.npy','query_indices.npy'}
    need(set(meta['artifact_sha256'])==artifacts and {p.name for p in folder.iterdir() if p.is_file()}==artifacts|{'metadata.json','metadata.sha256'},'Model inventory differs')
    for name,digest in meta['artifact_sha256'].items():add(folder/name,digest,stage+' model artifact')
    need(meta['round']==9 and meta['stage']==stage and meta['component']=='geography_poi_raw_catboost' and meta['family']=='catboost','Wrong model identity')
    need(meta['plan_sha256']==sha(OUT/'candidate_plan.json') and meta['source_sha256']==plan['source_sha256'],'Model plan/source differs')
    need(meta['frozen_candidate_sha256']==(None if stage=='development' else fsha),'Model freeze differs')
    need(meta['parameters']==plan['parameters'] and meta['iterations']==meta['actual_iterations']==2999 and meta['target_mode']=='raw' and meta['sample_weight']=='uniform','Model parameters/target differ')
    need(meta['feature_count']==109 and meta['categorical_feature_count']==23 and meta['feature_rules']==plan['feature_rules'],'Model feature schema differs')
    for key in ['target_formula','prediction_formula']:need(meta[key]==plan['feature_rules'][key],'Model formula differs')
    for key in ['target_used_for_geographic_state','query_used_for_geographic_state','reference_features_changed','external_property_answers_used','fit_uses_eval_set','early_stopping_used','query_labels_used']:need(meta[key] is False,'Model policy differs: '+key)
    need(meta['reload_64_rows_verified'] is True and meta['direct_component_score_omitted'] is True and meta['prediction_units']=='TWD','Model replay/units differ')
    need((meta['outer_fit_rows'],meta['query_rows'])==(spec['fit_rows'],spec['query_rows']) and meta['fit_rows']==spec['fit_rows'],'Model row counts differ')
    need(meta['geo_manifest_sha256']==plan['annual_geography'][stage]['geography_manifest_sha256'] and meta['poi_manifest_sha256']==sha(OUT/'poi_features'/stage/'manifest.json') and meta['poi_rules_sha256']==plan['poi_rules_sha256'],'Model map identity differs')
    state=read(folder/'feature_state.json');config=read(ROOT/'runs/v1/models/final_config.json')
    need(state['fit_state_uses_y'] is False and state['fit_state_uses_query'] is False and state['fit_rows']==spec['fit_rows'],'Feature state fit policy differs')
    need(state['feature_rules']==plan['feature_rules'] and state['parameters']==plan['parameters'] and state['source_sha256']==plan['source_sha256']['src/round9_geo_poi.py'],'Feature state rules differ')
    need(state['base_configuration_sha256']==meta['base_configuration_sha256']==sha(ROOT/'runs/v1/models/final_config.json'),'Base configuration differs')
    need(state['original_columns']==config['feature_columns'] and state['original_categorical_columns']==config['categorical_columns'],'Original schema differs')
    need(state['all_feature_names']==state['feature_columns']==meta['feature_columns'] and len(state['feature_columns'])==109 and len(state['categorical_columns'])==23 and state['categorical_columns']==meta['categorical_columns']==meta['categorical_features'],'Augmented schema differs')
    need(state['poi_columns']==plan['poi_rules']['columns'] and set(state['poi_dtypes'].values())=={'float32'},'POI schema differs')
    need(state['state_content_sha256']==meta['feature_state_content_sha256'],'State identity differs')
    for name,digest in state['dependency_source_sha256'].items():need(plan['source_sha256'].get(name)==digest,'State source differs')
    for prefix,short in [('original','X'),('geo','geo'),('poi','poi')]:
        key='fit_'+prefix+'_feature_values_sha256';need(state[key]==meta[key]==meta['fit_'+short+'_sha256'],'Fit feature identity differs')
    for role in ['fit','query']:
        indices=prior.verify_indices(folder/(role+'_indices.npy'),spec[role+'_rows'],spec[role+'_indices_content_sha256'])
        for name in plan['stage_dependency_sha256'][stage]:
            expected='fit_indices.npy' if role=='fit' else ('query_indices.npy','check_indices.npy')
            if Path(name).name==expected or (isinstance(expected,tuple) and Path(name).name in expected):need(np.array_equal(indices,np.load(ROOT/name,allow_pickle=False,mmap_mode='r')),'Prior row alignment differs')
    return meta

def verify_report(plan,frozen,accepted,results,metadata,manifests,add,mapping):
    report=read(ROOT/'reports/round9_results.json')
    need(report['round']==9 and report['candidate']==CANDIDATE and report['control']==CONTROL,'Report identity differs')
    need(report['accepted'] is accepted['accepted'] and report['current_released_recommendation']=='v3' and report['release_not_performed_by_this_report'] is True,'Report release/decision differs')
    for key in ['external_property_answers_used','public_score_used_for_selection','private_score_known','historically_untouched']:need(report[key] is False,'Report policy differs')
    need(report['formulas']==plan['formulas'] and report['thresholds']==plan['thresholds'],'Report rules differ')
    need(report['public_score_record']=={'submission_v2.csv':2883250.4138,'submission_v3.csv':None},'Report claims a new public score')
    need(report['unevaluated_stages']==accepted['unevaluated_stages'] and set(report['stages'])==set(results),'Report stage inventory differs')
    paths=[OUT/n for n in ['candidate_plan.json','candidate_plan.sha256','frozen_candidate.json','frozen_candidate.sha256','accepted_candidate.json','accepted_candidate.sha256']]
    paths += [ROOT/n for n in ['reports/round9_validation_plan.md','reports/round9_poi_design_review.md','reports/round9_model_design_review.md','outputs/round9/audit/preflight.json','outputs/round9/audit/poi_preparation_self_check.json','outputs/round9/audit/prepared_independent_review.json','src/round9_report.py']]
    for stage,result in results.items():
        model=metadata[stage];manifest=manifests[stage]
        expected={'metrics':result['metrics'],'decision':result['decision'],'fit_rows':result['fit_rows'],'query_rows':result['query_rows'],'snapshot_year':manifest['snapshot_year'],'source_pbf_sha256':manifest['source_pbf_sha256'],'replication_timestamp':manifest['replication_timestamp'],'geometry_records_by_class':manifest['geometry_records_by_class'],'feature_count':model['feature_count'],'categorical_feature_count':model['categorical_feature_count'],'fit_uses_eval_set':model['fit_uses_eval_set'],'early_stopping_used':model['early_stopping_used']}
        need(report['stages'][stage]==expected,'Report differs from saved stage: '+stage)
        paths += [OUT/'evaluation'/stage/'results.json',OUT/'evaluation'/stage/'results.sha256',OUT/'models'/stage/'metadata.json',OUT/'poi_features'/stage/'manifest.json',OUT/'poi_index'/str(YEARS[stage])/'audit.json']
    need(report['source_input_sha256']=={rel(p):sha(p) for p in paths},'Report source input identity/inventory differs')
    mapping(report['source_input_sha256'],'report verified source input')
    for path in REPORTS:add(path,reason='R9 report/document')

def contracts():
    need(sha(ROOT/'src/archive_round7.py')=='3e00d26f636167fe5a1f00aec734931807f7687cff7f2b3a04abceb7d4899951','Preserved archive helper changed')
    expected,reasons,add,mapping,locked=prior.registry()
    plan=locked(OUT/'candidate_plan.json','R9 frozen plan');frozen=locked(OUT/'frozen_candidate.json','R9 development freeze');accepted=locked(OUT/'accepted_candidate.json','R9 final decision')
    psha,fsha=sha(OUT/'candidate_plan.json'),sha(OUT/'frozen_candidate.json')
    need(plan['round']==9 and plan['candidate']==CANDIDATE and plan['control']==CONTROL and plan['single_development_candidate'] is True,'Wrong experiment')
    for key in ['tuning_or_weight_search','public_score_used_for_selection','historically_untouched','external_property_answers_used','fit_uses_eval_set','early_stopping_used','final_fit_implemented_in_this_driver']:need(plan[key] is False,'Plan policy differs')
    roots=[ROOT/'src/round9_experiment.py']+[ROOT/'src'/name for name in EXTRA_SOURCES]
    need(plan['source_sha256']=={rel(p):sha(p) for p in prior.source_closure(roots)} and len(plan['source_sha256'])==19,'Frozen source closure differs')
    mapping(plan['source_sha256'],'frozen R9 sources');mapping(plan['dependency_sha256'],'frozen dependencies')
    for stage,values in plan['stage_dependency_sha256'].items():mapping(values,stage+' old dependencies')
    mapping(plan['final_dependency_sha256'],'old final dependencies')
    old=locked(ROOT/'outputs/round6/candidate_plan.json','old R6 plan')
    for key in ['thresholds','splits','split_contract','target_group_weights','quarter_definition','final_contract']:need(plan[key]==old[key],'Old fixed contract differs: '+key)
    preflight=read(OUT/'audit/preflight.json')
    need(preflight['all_checks_passed'] is True and preflight['real_model_training_or_scoring'] is False and preflight['source_sha256']==plan['source_sha256'] and preflight['parameters']==plan['parameters'],'Preflight differs')
    for key in ['source_sha256','formulas','thresholds','parameters','poi_rules_sha256']:need(frozen[key]==plan[key],'Frozen rule differs')
    need(frozen['plan_sha256']==psha and frozen['historically_untouched'] is False,'Freeze identity differs')
    completed=completed_stages(frozen);mapping(frozen['development_artifact_sha256'],'frozen development artifacts')
    for kind in ['models','evaluation']:
        need({p.name for p in (OUT/kind).iterdir() if p.is_dir() and not p.name.startswith('.')}==set(completed),'Unauthorized or missing '+kind+' stage')
    need({p.name for p in (OUT/'poi_index').iterdir() if p.is_dir() and not p.name.startswith('.')}=={str(y) for y in YEARS.values()},'Annual catalog inventory differs')
    need({p.name for p in (OUT/'poi_features').iterdir() if p.is_dir() and not p.name.startswith('.')}==set(YEARS),'Annual feature inventory differs')
    manifests={s:verify_poi(s,plan,add,mapping,locked) for s in YEARS}
    results={};metadata={}
    for stage in completed:
        folder=OUT/'models'/stage;ef=OUT/'evaluation'/stage;spec=plan['split_contract'][stage]
        meta=verify_model(stage,plan,fsha,add,locked);r=locked(ef/'results.json',stage+' result')
        need(r['round']==9 and r['stage']==stage and r['plan_sha256']==psha and r['source_sha256']==plan['source_sha256'],'Result identity differs')
        need(r['frozen_candidate_sha256']==(None if stage=='development' else fsha),'Result freeze differs')
        for key in ['formulas','thresholds','target_group_weights']:need(r[key]==plan[key],'Result rules differ')
        need(r['evaluated_candidates']==[CONTROL,CANDIDATE] and set(r['metrics'])=={CONTROL,CANDIDATE},'Unexpected scored candidates')
        need((r['fit_rows'],r['query_rows'])==(spec['fit_rows'],spec['query_rows']),'Result row counts differ')
        need(r['v3_control_replayed_all_rows'] is True and r['new_model_replayed_all_rows'] is True,'Full-query replay proof absent')
        for key in ['candidate_component_separately_scored','query_labels_used_for_training','public_score_used','historically_untouched','new_public_score_known','private_score_known']:need(r[key] is False,'Result policy differs')
        validate_decision(stage,r['decision']);mapping(r['input_artifact_sha256'],stage+' scored input');mapping(r['output_artifact_sha256'],stage+' scored output')
        for name,digest in plan['stage_dependency_sha256'][stage].items():need(r['input_artifact_sha256'].get(name)==digest,'Result omits old stage dependency')
        for p in folder.iterdir():
            if p.is_file():need(r['input_artifact_sha256'].get(rel(p))==sha(p),'Result omits model artifact')
        for prefix in ['outputs/round9/poi_features/'+stage+'/','outputs/round9/poi_index/'+str(YEARS[stage])+'/']:
            for name,digest in expected.items():
                if name.startswith(prefix):need(r['input_artifact_sha256'].get(name)==digest,'Result omits POI dependency: '+name)
        need(r['output_artifact_sha256']=={rel(ef/(n+'.npy')):sha(ef/(n+'.npy')) for n in [CONTROL,CANDIDATE]},'Result prediction inventory differs')
        need({p.name for p in ef.iterdir() if p.is_file()}=={'results.json','results.sha256',CONTROL+'.npy',CANDIDATE+'.npy'},'Evaluation inventory differs')
        for p in [folder/'predictions.npy',ef/(CONTROL+'.npy'),ef/(CANDIDATE+'.npy')]:
            a=np.load(p,allow_pickle=False,mmap_mode='r');need(a.shape==(spec['query_rows'],) and a.dtype==np.dtype('float64') and np.isfinite(a).all() and (a>=0).all(),'Prediction vector schema differs')
        results[stage]=r;metadata[stage]=meta
    need(frozen['passed']==results['development']['decision']['passed'] and frozen['selected_candidate']==results['development']['decision']['selected_candidate'] and frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'Frozen result differs')
    need(frozen['development_artifact_sha256']=={rel(p):sha(p) for f in [OUT/'models/development',OUT/'evaluation/development'] for p in f.iterdir() if p.is_file()},'Frozen inventory differs')
    outcome=len(completed)==3 and all(r['decision']['passed'] for r in results.values())
    need(accepted['round']==9 and accepted['candidate']==CANDIDATE and accepted['accepted'] is outcome and accepted['plan_sha256']==psha and accepted['frozen_candidate_sha256']==fsha,'Final decision differs')
    need(accepted['deployment_decision']==(CANDIDATE if outcome else 'keep_v3') and accepted['no_reselection_or_parameter_change'] is True,'Final selection differs')
    need(accepted['result_sha256']=={s:sha(OUT/'evaluation'/s/'results.json') for s in completed} and accepted['unevaluated_stages']==[s for s in STAGES if s not in completed],'Final stage inventory differs')
    need(accepted['new_submission_created'] is False and accepted['private_score_known'] is False and accepted['source_sha256']==plan['source_sha256'],'Unexpected release/score')
    previous=prior.verify_prior(plan,add,locked);previous['experiments']=verify_prior_experiments(add,locked)
    for name in ['outputs/submission_v2.csv','outputs/submission_v3.csv']:add(name,plan['dependency_sha256'][name],'preserved submission')
    verify_report(plan,frozen,accepted,results,metadata,manifests,add,mapping)
    return plan,accepted,completed,previous,expected,reasons,add
def build_archive_plan():
    plan,accepted,completed,previous,expected,reasons,add=contracts()
    files=set()
    for p in OUT.rglob('*'):
        if any(part.startswith('.') or part=='__pycache__' for part in p.relative_to(OUT).parts):continue
        need(not p.is_symlink(),'Symlink not permitted in snapshot inputs')
        if p.is_file():
            need(p.suffix.lower() in {'.json','.jsonl','.sha256','.csv','.npy','.pkl','.txt','.md','.log','.cbm'},'Unclassified R9 artifact: '+rel(p));files.add(p.resolve())
    files.update((ROOT/name).resolve() for name in REPORTS)
    files.update(p.resolve() for p in (ROOT/'reports').glob('*round9*') if p.is_file())
    roots=set((ROOT/'src').glob('round9*.py'))|{Path(__file__).resolve()}
    for p in prior.source_closure(roots):
        if p.name.startswith('round9') or p.name=='archive_round9.py':files.add(p)
        else:add(p,reason='older imported source; external')
    for p in ROOT.glob('requirements*.txt'):add(p,reason='runtime requirements; external')
    originals={}
    for p in sorted(files):
        p=copy_allowed(rel(p));name=rel(p)
        originals[name]={'original_path':name,'original_absolute_path':str(p),'sha256':sha(p),'bytes':p.stat().st_size}
    external=[]
    for name,digest in sorted(expected.items()):
        if name in originals:need(originals[name]['sha256']==digest,'R9 file changed during collection')
        else:
            p=safe(ROOT,name);need(sha(p)==digest,'External changed during collection')
            external.append({'original_path':name,'original_absolute_path':str(p),'sha256':digest,'bytes':p.stat().st_size,'reasons':sorted(reasons[name])})
    return {'originals':originals,'external_dependencies':external,'completed_stages':completed,'accepted':accepted['accepted'],'acceptance_scope':'fixed historical backtests only; final fit/release required','selected_candidate':read(OUT/'frozen_candidate.json')['selected_candidate'],'deployment_decision':accepted['deployment_decision'],'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),'accepted_candidate_sha256':sha(OUT/'accepted_candidate.json'),'prior_archives_verified':previous,'new_release':False,'new_submission_created':False,'recommended_submission':'outputs/submission_v3.csv','checks':{'receipts_sources_inputs_outputs_verified':True,'model_artifact_provenance_verified':True,'fixed_partition_indices_verified':True,'four_annual_poi_and_geographic_closures_verified':True,'report_saved_results_equal':True,'prior_releases_experiments_registry_unchanged':True,'new_scores_computed':False}}

def verify_snapshot(folder):
    folder=safe(ROOT,folder);path=folder/MANIFEST
    need((folder/RECEIPT).read_text(encoding='ascii').strip()==sha(path),'Snapshot receipt differs')
    value=read(path);plan=value['original_plan']
    need(value['experiment_round']==9 and value['snapshot_kind']=='round9_experiments' and value['is_release'] is False,'Wrong snapshot identity')
    need(value['accepted']==plan['accepted'] and value['completed_stages']==plan['completed_stages'] and value['selected_candidate']==plan['selected_candidate'],'Snapshot outcome differs')
    for key in ['new_submission_created','versions_modified','prior_archives_modified','uploaded_to_kaggle']:need(value[key] is False,'Unexpected publication')
    need(value['acceptance_scope']==plan['acceptance_scope'] and value['self_contained'] is False,'Snapshot limitations differ')
    need(set(value['copied_files'])==set(plan['originals']),'Copy inventory differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(plan['originals'])|{MANIFEST,RECEIPT,NOTICE_FILE},'Snapshot inventory differs')
    for name,item in value['copied_files'].items():
        copy_allowed(name);original=plan['originals'][name]
        need(item['original_path']==name and item['backup_path']=='runs/round9_experiments/'+name,'Snapshot path differs')
        need(item['original_sha256']==item['backup_sha256']==original['sha256'] and item['original_bytes']==item['backup_bytes']==original['bytes'],'Copy proof differs')
        prior.verify_file(folder,name,item['backup_sha256'],item['backup_bytes'])
    need(value['external_dependencies']==plan['external_dependencies'],'External inventory differs')
    need((folder/NOTICE_FILE).read_bytes()==NOTICE.encode('utf-8'),'Notice differs')
    return value

def verify_current(plan):
    for name,item in plan['originals'].items():prior.verify_file(ROOT,name,item['sha256'],item['bytes'])
    for item in plan['external_dependencies']:prior.verify_file(ROOT,item['original_path'],item['sha256'],item['bytes'])
    need(build_archive_plan()==plan,'Contracts/sources/report/dependencies changed')

def self_check():
    checks=[]
    def ok(name,value):need(value,name);checks.append(name)
    def rejected(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    frozen={'passed':False,'later_stages_allowed':False,'selected_candidate':CONTROL}
    ok('rejected_development_only',completed_stages(frozen)==['development'])
    frozen.update(passed=True,later_stages_allowed=True,selected_candidate=CANDIDATE)
    ok('passed_development_requires_all_three',completed_stages(frozen)==list(STAGES))
    bad=dict(frozen,later_stages_allowed=False);rejected('inconsistent_authorization_rejected',lambda:completed_stages(bad))
    bad=dict(frozen,selected_candidate=CONTROL);rejected('wrong_candidate_rejected',lambda:completed_stages(bad))
    for stage in STAGES:
        keys={'large_strata_deteriorate_at_most_5pct'}
        if stage=='development':keys|={'overall_improves_at_least_1pct','weighted_improves_at_least_1pct','quarters_deteriorate_at_most_5pct'}
        else:
            keys|={'overall_within_limit','weighted_within_limit'}
            if stage=='confirmation':keys.add('at_least_two_quarters_improve')
        d={'checks':dict.fromkeys(keys,True),'passed':True,'selected_candidate':CANDIDATE,'candidate_reselection_performed':False,'improved_quarters':['Q1','Q2']}
        validate_decision(stage,d);checks.append(stage+'_saved_boolean_contract')
        bad=copy.deepcopy(d);bad['checks'][sorted(keys)[0]]=False
        rejected(stage+'_inconsistent_pass_rejected',lambda:validate_decision(stage,bad))
        bad['passed']=False;bad['selected_candidate']=CONTROL;validate_decision(stage,bad);checks.append(stage+'_legitimate_reject_supported')
    for name in ['outputs/round9/poi_index/2022/pois.jsonl','outputs/round9/poi_features/final/train.pkl','src/round9_poi_extract_annual.py','README_round9.md']:
        copy_allowed(name);checks.append('copy_scope_'+name)
    for name in ['outputs/submission_v3.csv','data_external/geography/taiwan-220101.osm.pbf','src/round6_geo.py','../outside.txt']:
        rejected('reject_copy_'+name,lambda n=name:copy_allowed(n))
    ok('accepted_notice_is_backtest_only','final fitting and release are still required' in NOTICE and 'not self-contained' in NOTICE)
    return {'all_checks_passed':True,'checks':checks,'real_model_training_or_scoring':False,'archive_created':False,'script_path':'src/archive_round9.py','script_sha256':sha(Path(__file__))}

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--verify',action='store_true');mode.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check()));return
    destination=safe(ROOT,DEST);runs=safe(ROOT,'runs')
    need(destination.parent==runs and destination.name=='round9_experiments','Unsafe destination')
    if args.verify:
        value=verify_snapshot(destination);verify_current(value['original_plan'])
        print(json.dumps({'status':'verified','snapshot':rel(destination),'files':len(value['copied_files']),'completed_stages':value['completed_stages'],'manifest_sha256':sha(destination/MANIFEST)}));return
    need(not destination.exists(),'Snapshot exists; use --verify; never overwrite')
    plan=build_archive_plan();runs.mkdir(parents=True,exist_ok=True);total=sum(item['bytes'] for item in plan['originals'].values())
    need(shutil.disk_usage(runs).free>total+16*1024*1024,'Insufficient disk space')
    staging=safe(runs,'.round9-staging-'+uuid.uuid4().hex);need(staging.parent==runs,'Unsafe staging');staging.mkdir();copied={}
    for name,item in sorted(plan['originals'].items()):
        source=safe(ROOT,name);target=safe(staging,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);prior.verify_file(staging,name,item['sha256'],item['bytes'])
        copied[name]={'original_path':name,'original_absolute_path':str(source),'original_sha256':item['sha256'],'original_bytes':item['bytes'],'backup_path':'runs/round9_experiments/'+name,'backup_sha256':sha(target),'backup_bytes':target.stat().st_size}
    (staging/NOTICE_FILE).write_bytes(NOTICE.encode('utf-8'))
    manifest={'format_version':1,'snapshot_kind':'round9_experiments','experiment_round':9,'is_release':False,'accepted':plan['accepted'],'acceptance_scope':plan['acceptance_scope'],'completed_stages':plan['completed_stages'],'selected_candidate':plan['selected_candidate'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'copied_files':copied,'external_dependencies':plan['external_dependencies'],'original_plan':plan,'self_contained':False,'dependency_notice':NOTICE_FILE,'new_submission_created':False,'versions_modified':False,'prior_archives_modified':False,'uploaded_to_kaggle':False,'manifest_self_excluded':True,'manifest_sha256_receipt':RECEIPT}
    (staging/MANIFEST).write_bytes(encoded(manifest));(staging/RECEIPT).write_text(sha(staging/MANIFEST)+'\n',encoding='ascii')
    verify_snapshot(staging);verify_current(plan)
    need(not destination.exists() and staging.parent==destination.parent==runs,'Unsafe publication or destination appeared')
    staging.rename(destination);verify_snapshot(destination)
    print(json.dumps({'status':'created','snapshot':rel(destination),'files':len(copied),'copied_bytes':total,'external_dependencies':len(plan['external_dependencies']),'completed_stages':plan['completed_stages'],'accepted_backtests_only':plan['accepted'],'manifest_sha256':sha(destination/MANIFEST),'versions_modified':False,'new_submission_created':False}))

if __name__=='__main__':main()