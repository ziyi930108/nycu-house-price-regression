"""Archive R8 decisions, references and reports; prior models/maps remain SHA-bound external files."""
from __future__ import annotations
import argparse,json,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import archive_round7 as prior
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round8';DEST=ROOT/'runs/round8_experiments'
STAGES=('development','confirmation','auxiliary')
CONTROL='v3_control';CANDIDATE='project_balanced_spatial_reference'
MANIFEST='snapshot_manifest.json';RECEIPT='snapshot_manifest.sha256';NOTICE_FILE='SNAPSHOT_README.md'
REPORTS=('README_round8.md','reports/optimization_round8_report.md','reports/round8_results.json','reports/round8_work_log.md','reports/round8_validation_plan.md','reports/round8_design_review.md','reports/round8_location_feasibility.md')
NOTICE='''# R8 finalized experiment snapshot

This preserves the R8 reference construction, backtest decisions, receipts,
source files, audits, work log and report. It does not publish a new submission.
Previous models, sources, input spreadsheets, feature caches, geographic maps and
release snapshots remain SHA-bound external dependencies at their original paths.
The snapshot is not self-contained. An accepted historical backtest is not an
independent private-score guarantee. No new Public or Private score is claimed.
The snapshot manifest excludes itself and has a separate SHA256 receipt.
'''
need,sha,read,rel,safe,encoded=prior.need,prior.sha,prior.read,prior.rel,prior.safe,prior.encoded

def contracts():
    need(sha(ROOT/'src/archive_round7.py')=='3e00d26f636167fe5a1f00aec734931807f7687cff7f2b3a04abceb7d4899951','Preserved archive helper changed')
    expected,reasons,add,mapping,locked=prior.registry()
    plan=locked(OUT/'candidate_plan.json','R8 frozen plan')
    frozen=locked(OUT/'frozen_candidate.json','R8 development decision')
    accepted=locked(OUT/'accepted_candidate.json','R8 final decision')
    psha,fsha=sha(OUT/'candidate_plan.json'),sha(OUT/'frozen_candidate.json')
    need(plan['round']==8 and plan['candidate']==CANDIDATE and plan['control']==CONTROL,'Wrong experiment')
    need(plan['single_development_candidate'] is True and plan['new_tree_models_trained']==0,'Wrong candidate scope')
    for flag in ['tuning_or_weight_search','public_score_used_for_selection','historically_untouched','external_property_answers_used','query_labels_passed_to_component','query_X_used_to_fit_reference']:need(plan[flag] is False,'Plan policy differs: '+flag)
    need(plan['source_sha256']=={rel(p):sha(p) for p in prior.source_closure([ROOT/'src/round8_experiment.py'])},'Frozen experiment source closure differs')
    mapping(plan['source_sha256'],'frozen R8 source closure');mapping(plan['dependency_sha256'],'frozen input/research')
    for stage,values in plan['stage_dependency_sha256'].items():mapping(values,stage+' old inputs')
    mapping(plan['final_dependency_sha256'],'old final inputs')
    old=locked(ROOT/'outputs/round6/candidate_plan.json','preserved R6 contract')
    for key in ['thresholds','splits','split_contract','target_group_weights','quarter_definition','final_contract']:need(plan[key]==old[key],'Prior fixed contract differs: '+key)
    preflight=read(OUT/'audit/preflight.json')
    need(preflight['all_checks_passed'] is True and preflight['real_model_training_or_scoring'] is False and preflight['source_sha256']==plan['source_sha256'] and preflight['parameters']==plan['parameters'],'Preflight differs')
    for key in ['source_sha256','formulas','thresholds','parameters']:need(frozen[key]==plan[key],'Freeze rule differs')
    need(frozen['plan_sha256']==psha and type(frozen['passed']) is bool and frozen['later_stages_allowed']==frozen['passed'],'Freeze authorization differs')
    mapping(frozen['development_artifact_sha256'],'frozen development files')
    completed=list(STAGES) if frozen['passed'] else ['development']
    for kind in ['models','evaluation']:
        actual={p.name for p in (OUT/kind).iterdir() if p.is_dir() and not p.name.startswith('.')}
        need(actual==set(completed),'Unauthorized or missing stage: '+kind)
    results={};metadata={}
    for stage in completed:
        sf=OUT/'models'/stage;ef=OUT/'evaluation'/stage;spec=plan['split_contract'][stage]
        m=locked(sf/'metadata.json',stage+' reference metadata');r=locked(ef/'results.json',stage+' result')
        for value in [m,r]:
            need(value['round']==8 and value['stage']==stage and value['plan_sha256']==psha and value['source_sha256']==plan['source_sha256'],'Stage identity differs')
            need(value['frozen_candidate_sha256']==(None if stage=='development' else fsha),'Stage freeze differs')
        need(m['component']==CANDIDATE and m['parameters']==plan['parameters'] and m['new_tree_models_trained']==0,'Reference metadata differs')
        for key in ['query_labels_used','reference_state_uses_query']:need(m[key] is False,'Reference leakage contract differs')
        for key in ['reload_64_rows_verified','v3_control_replayed_all_rows','feature_frames_unchanged']:need(m[key] is True,'Replay/invariance proof absent')
        need(m['prediction_units']=='TWD' and (m['outer_fit_rows'],m['query_rows'])==(spec['fit_rows'],spec['query_rows']),'Reference units/rows differ')
        need(m['geo_manifest_sha256']==plan['stage_dependency_sha256'][stage]['outputs/round6/geography/'+stage+'/manifest.json'],'Wrong map version')
        need(m['reference_sha256']==plan['stage_dependency_sha256'][stage][m['reference_path']],'Wrong old fit-only reference')
        artifacts={'reference.pkl','state_summary.json','prediction_report.json','diagnostics.pkl','fit_indices.npy','query_indices.npy','predictions.npy'}
        need(set(m['artifact_sha256'])==artifacts and {p.name for p in sf.iterdir() if p.is_file()}==artifacts|{'metadata.json','metadata.sha256'},'Reference inventory differs')
        for name,digest in m['artifact_sha256'].items():add(sf/name,digest,stage+' reference artifact')
        for role in ['fit','query']:
            indices=prior.verify_indices(sf/(role+'_indices.npy'),spec[role+'_rows'],spec[role+'_indices_content_sha256'])
            old_name='fit_indices.npy' if role=='fit' else 'query_indices.npy'
            need(np.array_equal(indices,np.load(ROOT/'outputs/round6/models'/stage/'geography_catboost'/old_name,allow_pickle=False)),'Prior row alignment differs')
        for key in ['formulas','thresholds','target_group_weights']:need(r[key]==plan[key],'Scored rules differ')
        need(r['evaluated_candidates']==[CONTROL,CANDIDATE] and set(r['metrics'])=={CONTROL,CANDIDATE},'Unexpected scored candidates')
        need((r['fit_rows'],r['query_rows'])==(spec['fit_rows'],spec['query_rows']),'Scored rows differ')
        need(r['v3_control_replayed_all_rows'] is True and r['new_reference_replayed_all_rows'] is True,'All-row replay proof absent')
        for key in ['query_labels_used_for_training','public_score_used','historically_untouched','new_public_score_known','private_score_known']:need(r[key] is False,'Scoring policy differs')
        d=r['decision'];need(d['passed']==all(d['checks'].values()) and all(type(v) is bool for v in d['checks'].values()),'Gate flags differ')
        need(d['selected_candidate']==(CANDIDATE if d['passed'] else CONTROL) and d['candidate_reselection_performed'] is False,'Selection differs')
        mapping(r['input_artifact_sha256'],stage+' scored inputs');mapping(r['output_artifact_sha256'],stage+' scored outputs')
        for name,digest in plan['stage_dependency_sha256'][stage].items():need(r['input_artifact_sha256'].get(name)==digest,'Scoring omits stage dependency')
        for path in sf.iterdir():
            if path.is_file():need(r['input_artifact_sha256'].get(rel(path))==sha(path),'Scoring omits reference artifact')
        need(r['output_artifact_sha256']=={rel(ef/(n+'.npy')):sha(ef/(n+'.npy')) for n in [CONTROL,CANDIDATE]},'Prediction inventory differs')
        need({p.name for p in ef.iterdir() if p.is_file()}=={'results.json','results.sha256',CONTROL+'.npy',CANDIDATE+'.npy'},'Evaluation file inventory differs')
        for path in [sf/'predictions.npy',ef/(CONTROL+'.npy'),ef/(CANDIDATE+'.npy')]:
            a=np.load(path,allow_pickle=False,mmap_mode='r');need(a.shape==(spec['query_rows'],) and a.dtype==np.dtype('float64') and np.isfinite(a).all() and (a>=0).all(),'Invalid saved prediction vector')
        results[stage]=r;metadata[stage]=m
    need(frozen['passed']==results['development']['decision']['passed'] and frozen['selected_candidate']==results['development']['decision']['selected_candidate'] and frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'Frozen result differs')
    need(frozen['development_artifact_sha256']=={rel(p):sha(p) for f in [OUT/'models/development',OUT/'evaluation/development'] for p in f.iterdir() if p.is_file()},'Frozen inventory differs')
    outcome=len(completed)==3 and all(r['decision']['passed'] for r in results.values())
    need(accepted['round']==8 and accepted['candidate']==CANDIDATE and accepted['accepted']==outcome and accepted['plan_sha256']==psha and accepted['frozen_candidate_sha256']==fsha,'Final decision differs')
    need(accepted['deployment_decision']==(CANDIDATE if outcome else 'keep_v3') and accepted['no_reselection_or_parameter_change'] is True,'Final deployment differs')
    need(accepted['result_sha256']=={s:sha(OUT/'evaluation'/s/'results.json') for s in completed} and accepted['unevaluated_stages']==[s for s in STAGES if s not in completed],'Final stage inventory differs')
    need(accepted['new_submission_created'] is False and accepted['private_score_known'] is False and accepted['source_sha256']==plan['source_sha256'],'Unexpected release/score')
    previous=prior.verify_prior(plan,add,locked)
    report=read(ROOT/'reports/round8_results.json')
    for name in REPORTS:add(name,reason='R8 report')
    need(report['round']==8 and report['metric']=='MAE' and report['unit']=='TWD','Report identity differs')
    need(report['plan_sha256']==psha and report['frozen_candidate_sha256']==fsha and report['accepted_candidate_sha256']==sha(OUT/'accepted_candidate.json'),'Report contracts differ')
    need(report['acceptance']==accepted and report['acceptance_finalized'] is True and report['backtest_accepted']==outcome,'Report decision differs')
    need(report['selected_candidate']==frozen['selected_candidate'],'Report frozen selection differs')
    for key in ['formulas','thresholds','parameters']:need(report[key]==plan[key],'Report rules differ: '+key)
    need(report['recommendation']=='v3' and report['recommended_submission']=='outputs/submission_v3.csv' and report['new_submission'] is None and report['new_release_verified'] is False,'Experiment report claims release')
    need(report['public_mae'] is None and report['private_mae'] is None and report['historically_untouched'] is False,'Report claims unknown score or independent holdout')
    need(set(report['stages'])==set(STAGES),'Report stage inventory differs')
    for stage in STAGES:
        item=report['stages'][stage];spec=plan['split_contract'][stage]
        result_path=OUT/'evaluation'/stage/'results.json';metadata_path=OUT/'models'/stage/'metadata.json'
        need(item['dates']==plan['splits'][stage] and (item['fit_rows'],item['query_rows'])==(spec['fit_rows'],spec['query_rows']),'Report stage contract differs')
        need(item['result_path']==rel(result_path) and item['metadata_path']==rel(metadata_path),'Report stage artifact paths differ')
        if stage in results:
            need(item['status']=='evaluated' and item['result_sha256']==sha(result_path) and item['metadata_sha256']==sha(metadata_path),'Report completed artifact identity differs')
            need(item['metrics']==results[stage]['metrics'] and item['decision']==results[stage]['decision'],'Report differs from result')
        else:need(item['status']=='not_required' and item['metrics'] is None and item['decision'] is None and item['result_sha256'] is None and item['metadata_sha256'] is None,'Unexecuted stage falsely reported')
    mapping(report['receipt_verified_inputs'],'report receipt input');mapping(report['document_sha256'],'report document')
    add('src/round8_report.py',report['report_script_sha256'],'report generator')
    return plan,accepted,completed,previous,expected,reasons,add

def build_archive_plan():
    plan,accepted,completed,previous,expected,reasons,add=contracts()
    files=set()
    for path in OUT.rglob('*'):
        if path.is_file() and not path.is_symlink() and not any(part.startswith('.') or part=='__pycache__' for part in path.relative_to(OUT).parts):
            need(path.suffix.lower() in {'.json','.sha256','.csv','.npy','.pkl','.txt','.md','.log'},'Unclassified R8 artifact: '+rel(path));files.add(path.resolve())
    files.update((ROOT/name).resolve() for name in REPORTS)
    files.update(p.resolve() for p in (ROOT/'reports').glob('*round8*') if p.is_file())
    roots=set((ROOT/'src').glob('round8*.py'))|{Path(__file__).resolve()}
    for path in prior.source_closure(roots):
        if path.name.startswith('round8') or path.name=='archive_round8.py':files.add(path)
        else:add(path,reason='older imported source; external')
    for path in ROOT.glob('requirements*.txt'):add(path,reason='runtime requirements; external')
    originals={}
    for path in sorted(files):
        path=safe(ROOT,path);name=rel(path)
        need(name.startswith(('outputs/round8/','reports/','src/')) or name=='README_round8.md','Wrong archive copy scope')
        if name.startswith('src/'):need(path.name.startswith('round8') or path.name=='archive_round8.py','Cannot copy older source')
        if name.startswith('reports/'):need('round8' in path.name,'Cannot copy older report')
        originals[name]={'original_path':name,'original_absolute_path':str(path),'sha256':sha(path),'bytes':path.stat().st_size}
    external=[]
    for name,digest in sorted(expected.items()):
        if name in originals:need(originals[name]['sha256']==digest,'Current-round file changed during collection')
        else:
            path=safe(ROOT,name);need(sha(path)==digest,'External changed during collection')
            external.append({'original_path':name,'original_absolute_path':str(path),'sha256':digest,'bytes':path.stat().st_size,'reasons':sorted(reasons[name])})
    return {'originals':originals,'external_dependencies':external,'completed_stages':completed,'accepted':accepted['accepted'],'selected_candidate':read(OUT/'frozen_candidate.json')['selected_candidate'],'deployment_decision':accepted['deployment_decision'],'plan_sha256':sha(OUT/'candidate_plan.json'),'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),'accepted_candidate_sha256':sha(OUT/'accepted_candidate.json'),'prior_archives_verified':previous,'new_release':False,'new_submission_created':False,'recommended_submission':'outputs/submission_v3.csv','checks':{'receipts_sources_inputs_outputs_verified':True,'reference_artifact_provenance_verified':True,'fixed_partition_indices_verified':True,'report_saved_results_equal':True,'prior_releases_and_registry_unchanged':True,'new_scores_computed':False}}

def verify_snapshot(folder):
    folder=safe(ROOT,folder);path=folder/MANIFEST
    need((folder/RECEIPT).read_text(encoding='ascii').strip()==sha(path),'Snapshot receipt differs')
    value=read(path);plan=value['original_plan']
    need(value['experiment_round']==8 and value['snapshot_kind']=='round8_experiments' and value['is_release'] is False,'Wrong snapshot identity')
    need(value['accepted']==plan['accepted'] and value['completed_stages']==plan['completed_stages'],'Snapshot outcome differs')
    need(set(value['copied_files'])==set(plan['originals']),'Copy inventory differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==set(plan['originals'])|{MANIFEST,RECEIPT,NOTICE_FILE},'Snapshot inventory differs')
    for name,item in value['copied_files'].items():
        original=plan['originals'][name]
        need(item['original_path']==name and item['backup_path']=='runs/round8_experiments/'+name,'Snapshot path differs')
        need(item['original_sha256']==item['backup_sha256']==original['sha256'] and item['original_bytes']==item['backup_bytes']==original['bytes'],'Copy proof differs')
        prior.verify_file(folder,name,item['backup_sha256'],item['backup_bytes'])
    need(value['external_dependencies']==plan['external_dependencies'],'External inventory differs')
    need((folder/NOTICE_FILE).read_bytes()==NOTICE.encode('utf-8'),'Notice differs')
    return value

def verify_current(plan):
    for name,item in plan['originals'].items():prior.verify_file(ROOT,name,item['sha256'],item['bytes'])
    for item in plan['external_dependencies']:prior.verify_file(ROOT,item['original_path'],item['sha256'],item['bytes'])
    need(build_archive_plan()==plan,'Contracts/sources/report/dependencies changed')

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    destination=safe(ROOT,DEST);runs=safe(ROOT,'runs')
    need(destination.parent==runs and destination.name=='round8_experiments','Unsafe archive destination')
    if args.verify:
        value=verify_snapshot(destination);verify_current(value['original_plan'])
        print(json.dumps({'status':'verified','snapshot':rel(destination),'files':len(value['copied_files']),'completed_stages':value['completed_stages'],'manifest_sha256':sha(destination/MANIFEST)}));return
    need(not destination.exists(),'Snapshot exists; use --verify; never overwrite')
    plan=build_archive_plan();runs.mkdir(parents=True,exist_ok=True);total=sum(item['bytes'] for item in plan['originals'].values())
    need(shutil.disk_usage(runs).free>total+16*1024*1024,'Insufficient disk space')
    staging=safe(runs,'.round8-staging-'+uuid.uuid4().hex);need(staging.parent==runs,'Unsafe staging');staging.mkdir()
    copied={}
    for name,item in sorted(plan['originals'].items()):
        source=safe(ROOT,name);target=safe(staging,name);target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target);prior.verify_file(staging,name,item['sha256'],item['bytes'])
        copied[name]={'original_path':name,'original_absolute_path':str(source),'original_sha256':item['sha256'],'original_bytes':item['bytes'],'backup_path':'runs/round8_experiments/'+name,'backup_sha256':sha(target),'backup_bytes':target.stat().st_size}
    (staging/NOTICE_FILE).write_bytes(NOTICE.encode('utf-8'))
    manifest={'format_version':1,'snapshot_kind':'round8_experiments','experiment_round':8,'is_release':False,'accepted':plan['accepted'],'completed_stages':plan['completed_stages'],'selected_candidate':plan['selected_candidate'],'created_at_utc':datetime.now(timezone.utc).isoformat(),'copied_files':copied,'external_dependencies':plan['external_dependencies'],'original_plan':plan,'self_contained':False,'dependency_notice':NOTICE_FILE,'new_submission_created':False,'versions_modified':False,'prior_archives_modified':False,'uploaded_to_kaggle':False,'manifest_self_excluded':True,'manifest_sha256_receipt':RECEIPT}
    (staging/MANIFEST).write_bytes(encoded(manifest));(staging/RECEIPT).write_text(sha(staging/MANIFEST)+'\n',encoding='ascii')
    verify_snapshot(staging);verify_current(plan)
    need(not destination.exists() and staging.parent==destination.parent==runs,'Unsafe publication or destination appeared')
    staging.rename(destination);verify_snapshot(destination)
    print(json.dumps({'status':'created','snapshot':rel(destination),'files':len(copied),'copied_bytes':total,'external_dependencies':len(plan['external_dependencies']),'completed_stages':plan['completed_stages'],'accepted':plan['accepted'],'manifest_sha256':sha(destination/MANIFEST),'versions_modified':False,'new_submission_created':False}))

if __name__=='__main__':main()
