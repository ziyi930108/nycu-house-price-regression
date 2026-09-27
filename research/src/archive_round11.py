"""Archive finalized R11 backtests only; no fitting, prediction, scoring or release.
Default creates runs/round11_experiments once. --verify checks the immutable
snapshot, its frozen originals and exact external dependencies. --self-check
uses only synthetic in-memory contracts and creates no files.
"""
from __future__ import annotations
import argparse,copy,hashlib,json,re,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path,PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round11';DEST=ROOT/'runs/round11_experiments'
STAGES=('development','confirmation','auxiliary')
CONTROL='v3_control';CANDIDATE='importance_geo_blend'
MUTABLE={'CURRENT_STATUS.md','runs/versions.json'}
NOTICE='''# R11 finalized backtest snapshot

This is an experiment snapshot, not a model release. Even accepted=true means
only that the fixed historical backtests passed. The released recommendation
remains outputs/submission_v3.csv. No submission or registry entry is created.

Only R11 sources, reports and successful outputs are copied. Original feature
caches, supplied data, old models, old map/feature manifests and their complete
source/index dependencies are SHA-bound external dependencies. This snapshot
is not self-contained. Mutable status and versions entry points are not copied
or bound as live originals. Their recorded plan-time values remain historical.

Verification reads bytes, indices and saved JSON contracts, never property
labels or price-prediction values, and never refits or recalculates row MAE.
Incomplete staging is rejected and preserved; nothing is recursively deleted.
'''

def need(ok,message):
    if not ok:raise RuntimeError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def normalized(name):
    name=str(name).replace('\\','/');p=PurePosixPath(name)
    need(name and not p.is_absolute() and ':' not in name and '..' not in p.parts and '.' not in p.parts,'Unsafe relative path: '+name)
    return p.as_posix()
def safe(name,base=ROOT,exists=True):
    name=normalized(name);p=(base/name).resolve();need(p.is_relative_to(base.resolve()),'Path escapes workspace/snapshot: '+name)
    if exists:need(p.is_file(),'Missing file: '+str(p))
    return p
def rel(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def locked(path):
    path=Path(path);need(path.with_suffix('.sha256').read_text(encoding='ascii').strip()==sha(path),'Receipt mismatch: '+str(path));return read(path)
def value_sha(a):
    import numpy as np
    return hashlib.sha256(np.asarray(a,dtype='<i8').tobytes()).hexdigest()
def content(value):return hashlib.sha256(json.dumps(value,ensure_ascii=False,sort_keys=True,allow_nan=False).encode()).hexdigest()

class Bindings:
    def __init__(self):self.values={};self.case_keys={};self.mutable_observations={}
    def observe_mutable(self,name,digest):
        name=normalized(name);need(name in MUTABLE and re.fullmatch("[0-9a-f]{64}",digest),"Invalid historical context")
        need(name not in self.mutable_observations or self.mutable_observations[name]==digest,"Conflicting historical context")
        self.mutable_observations[name]=digest
    def declare(self,name,digest):
        name=normalized(name);need(name not in MUTABLE and name.casefold() not in {x.casefold() for x in MUTABLE},'Mutable entry point cannot be frozen: '+name)
        need(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Invalid SHA: '+name)
        prior=self.case_keys.get(name.casefold());need(prior is None or prior==name,'Case-ambiguous dependency: '+name)
        need(name not in self.values or self.values[name]==digest,'Conflicting SHA for '+name)
        self.case_keys[name.casefold()]=name;self.values[name]=digest
    def add(self,name,digest=None):
        path=safe(name);name=rel(path);actual=sha(path);need(digest is None or digest==actual,'Recorded SHA changed: '+name)
        if name in MUTABLE:self.observe_mutable(name,actual)
        else:self.declare(name,actual)
        return path
    def mapping(self,mapping):
        need(isinstance(mapping,dict),'Expected path-to-SHA map')
        for name,digest in mapping.items():self.add(name,digest)
    def authority(self,name):
        p=self.add(name);v=locked(p);self.add(rel(p.with_suffix('.sha256')));return v
    def recheck(self):
        for name,digest in self.values.items():need(sha(safe(name))==digest,'Dependency changed during archive: '+name)

def reject_staging_names(names):
    bad=[]
    for name in names:
        parts=PurePosixPath(normalized(name)).parts
        if any(p.startswith(('.', '_')) or p.lower().endswith(('.tmp','.part','.lock')) for p in parts):bad.append(name)
    need(not bad,'Incomplete/staging paths must be inspected, not skipped/deleted: '+', '.join(bad))

def fixed_finalization(frozen,accepted,results,weights,model_stages):
    need(set(results)<=set(STAGES) and 'development' in results and set(weights)<=set(STAGES),'Invalid completed stage inventory')
    need(type(frozen['passed']) is bool and frozen['later_stages_allowed'] is frozen['passed'],'Invalid freeze permission')
    need(frozen['selected_candidate']==(CANDIDATE if frozen['passed'] else CONTROL),'Wrong frozen selection')
    need(results['development']['decision']['passed'] is frozen['passed'],'Freeze/development decision differs')
    need('development' in weights and weights['development']['feasibility_passed'] is True,'Development weights not passed')
    need(set(model_stages)==set(results),'Incomplete model/result inventory')
    statuses={'development':'price_evaluated'};failed=[]
    for stage in STAGES[1:]:
        if not frozen['passed']:
            need(stage not in weights and stage not in results,'Rejected development has later work')
            statuses[stage]='not_executed_after_development_rejection'
        elif stage not in weights:
            need(stage not in results,'Price score missing preparation');statuses[stage]='not_executed'
        elif weights[stage]['feasibility_passed'] is False:
            need(stage not in results,'Failed pure-X stage has a price score');failed.append(stage);statuses[stage]='pure_X_feasibility_failed_price_not_evaluated'
        elif stage in results:statuses[stage]='price_evaluated'
        else:statuses[stage]='weights_prepared_price_not_executed'
    if not frozen['passed']:ok=False;reason='development_price_gates_failed'
    elif failed:
        ok=False;reason='pure_X_feasibility_failed'
        statuses={s:('not_executed_after_pure_X_stop' if v=='not_executed' else v) for s,v in statuses.items()}
    else:
        need(set(results)==set(STAGES),'Missing later results are incomplete execution, not rejection')
        ok=all(v['decision']['passed'] for v in results.values());reason='three_stage_price_gates_passed' if ok else 'later_price_gates_failed'
    need(accepted['accepted'] is ok and accepted['stopping_reason']==reason,'Final outcome differs')
    need(accepted['stage_status']==statuses and accepted['pure_X_feasibility_failure_stages']==failed,'Final stage states differ')
    need(accepted['unevaluated_stages']==[s for s in STAGES if s not in results],'Unevaluated inventory differs')
    need(accepted['selected_candidate']==frozen['selected_candidate'],'Final candidate differs from freeze')
    need(accepted['deployment_decision']==('backtest_accepted_pending_final_release' if ok else 'keep_v3'),'Unexpected deployment claim')
    need(accepted['new_submission_created'] is False and accepted['currently_recommended_submission']=='outputs/submission_v3.csv','Unexpected release/submission')
    return statuses,failed

def indices(path,rows,digest):
    import numpy as np
    a=np.load(path,allow_pickle=False)
    need(a.ndim==1 and np.issubdtype(a.dtype,np.integer) and len(a)==rows and len(np.unique(a))==rows and (a>=0).all(),'Invalid saved indices')
    need(value_sha(a)==digest,'Saved index content differs');return a

def check_model(stage,plan,weights,binding,driver):
    import numpy as np
    folder=OUT/'models'/stage;meta=binding.authority(rel(folder/'metadata.json'));state=read(folder/'feature_state.json')
    expected_files={'model.cbm','feature_state.json','predictions.npy','fit_indices.npy','query_indices.npy'}
    need(set(meta['artifact_sha256'])==expected_files,'Model artifact inventory differs')
    need({p.name for p in folder.iterdir() if p.is_file()}==expected_files|{'metadata.json','metadata.sha256'},'Unexpected model artifacts')
    for name,digest in meta['artifact_sha256'].items():binding.add(rel(folder/name),digest)
    binding.mapping(meta['source_sha256']);binding.mapping(meta['weight_preparation_input_sha256'])
    spec=plan['split_contract'][stage];wpath=OUT/'weights'/stage/'manifest.json'
    contract={'phase':stage,**{k:weights[k] for k in ['fit_X_sha256','query_X_sha256','fit_indices_content_sha256','query_indices_content_sha256']},'weights_values_sha256':weights['fit_weight_values_sha256'],'domain_plan_sha256':plan['domain_plan_sha256'],'preparation_manifest_sha256':sha(wpath),'query_X_used_for_weights':True,'query_labels_used_for_weights':False}
    driver.validate_model_contract(meta,plan,{'stage':stage,'weight_contract':contract,'plan_sha256':sha(OUT/'candidate_plan.json'),'domain_plan_sha256':plan['domain_plan_sha256'],'preparation_manifest_sha256':sha(wpath),'source_sha256':plan['source_sha256']})
    need(meta['frozen_candidate_sha256']==(None if stage=='development' else sha(OUT/'frozen_candidate.json')),'Model freeze identity differs')
    need(meta['outer_fit_rows']==spec['fit_rows'] and meta['query_rows']==spec['query_rows'],'Model row counts differ')
    need(meta['reload_64_rows_verified'] is True and meta['direct_component_score_omitted'] is True and meta['prediction_units']=='TWD','Model replay/unit contract differs')
    need(meta['geo_manifest_sha256']==sha(ROOT/'outputs/round6/geography'/stage/'manifest.json'),'Model geographic version differs')
    need(state['state_content_sha256']==content({k:v for k,v in state.items() if k!='state_content_sha256'})==meta['feature_state_content_sha256'],'Model feature state identity differs')
    need(state['weight_contract']==contract and state['weight_contract_sha256']==content(contract)==meta['weight_contract_sha256'],'Stored training weight contract differs')
    need(state['fit_rows']==spec['fit_rows'] and state['fit_sample_weight_values_sha256']==meta['fit_sample_weight_values_sha256']==weights['fit_weight_values_sha256'],'Stored weight values differ')
    need(state['feature_rules']==plan['feature_rules'] and state['parameters']==plan['parameters'] and len(state['all_feature_names'])==95 and len(state['categorical_columns'])==23,'State feature/parameter schema differs')
    need(state['fit_state_uses_y'] is False and state['geographic_feature_state_uses_query'] is False and state['query_X_used_for_training_weights'] is True and state['query_labels_used'] is False,'State data-use contract differs')
    geo_state=state['geo_state']
    need(geo_state['state_content_sha256']==content({k:v for k,v in geo_state.items() if k!='state_content_sha256'})==meta['geo_state_content_sha256'],'Nested geography state changed')
    need(geo_state['fit_original_feature_values_sha256']==meta['fit_X_sha256']==weights['fit_X_sha256'] and geo_state['fit_geo_feature_values_sha256']==meta['fit_geo_sha256'],'Nested geographic fit identity differs')
    need(geo_state['fit_rows']==spec['fit_rows'] and geo_state['fit_state_uses_y'] is False and geo_state['fit_state_uses_query'] is False,'Nested geographic data-use differs')
    binding.mapping(state['source_sha256']);binding.add('runs/v1/models/final_config.json',state['base_configuration_sha256'])
    for role in ('fit','query'):
        a=indices(folder/(role+'_indices.npy'),spec[role+'_rows'],spec[role+'_indices_content_sha256'])
        need(meta[role+'_indices_content_sha256']==spec[role+'_indices_content_sha256'] and meta[role+'_X_sha256']==weights[role+'_X_sha256'],'Model/preparation rows differ')
        old=ROOT/'outputs/round6/models'/stage/'geography_catboost'/(role+'_indices.npy')
        need(np.array_equal(a,np.load(old,allow_pickle=False)),'Model indices differ from v3')
    return meta

def validate_report(report,plan,frozen,accepted,results,weights,models,binding):
    need(report['round']==11 and report['candidate']==CANDIDATE and report['control']==CONTROL,'Report identity differs')
    need(report['metric']=='MAE' and report['unit']=='TWD','Report metric/units differ')
    need(report['accepted'] is accepted['accepted'] and report['selected_candidate']==accepted['selected_candidate'] and report['stopping_reason']==accepted['stopping_reason'],'Report final decision differs')
    need(report['status']==('backtest_accepted_pending_release' if accepted['accepted'] else 'rejected_keep_v3'),'Report status differs')
    need(report['current_released_recommendation']=='v3' and report['recommended_submission']=='outputs/submission_v3.csv','Report recommends an unissued release')
    for key in ['new_submission_created','release_performed','kaggle_upload_performed','query_labels_used_for_training','training_importance_weights_used_for_scoring','external_property_answers_used','public_score_used_for_selection','historically_untouched','private_score_known']:
        need(report[key] is False,'Report policy differs: '+key)
    need(report['query_X_used_for_training_weights'] is True and report['private_mae'] is None,'Report transductive/private declaration differs')
    need(report['public_score_record']=={'submission_v2.csv':2883250.4138,'submission_v3.csv':None,'round11_candidate':None},'Report invents Public evidence')
    for key in ['formulas','thresholds','parameters','domain_config','feature_rules','target_group_weights']:need(report[key]==plan[key],'Report rules differ: '+key)
    authorities={key:sha(OUT/name) for key,name in [('plan_sha256','candidate_plan.json'),('domain_plan_sha256','domain_plan.json'),('frozen_candidate_sha256','frozen_candidate.json'),('accepted_candidate_sha256','accepted_candidate.json')]}
    for key,digest in authorities.items():need(report[key]==digest,'Report authority differs')
    need(report['report_path']=='reports/round11_report.md' and report['report_markdown_sha256']==sha(ROOT/report['report_path']),'Report markdown changed')
    need(report['script_path']=='src/round11_report.py' and report['script_sha256']==sha(ROOT/report['script_path']),'Report generator source changed')
    binding.mapping(report['source_input_sha256'])
    # These are pure formatting/contract functions: never call report.generate/load_verified.
    import round11_report as reporter
    need(set(report['stages'])==set(STAGES) and report['unevaluated_stages']==accepted['unevaluated_stages'],'Report stage inventory differs')
    for stage,row in report['stages'].items():
        need(row['metrics']==(results[stage]['metrics'] if stage in results else None) and row['decision']==(results[stage]['decision'] if stage in results else None),'Report stage metrics/decision differs')
        expected=reporter.make_stage(plan,stage,accepted['stage_status'][stage],results.get(stage),models.get(stage),weights.get(stage))
        for field in ['result','model_metadata','weight_manifest']:
            if expected[field+'_path'] is not None:expected[field+'_sha256']=sha(safe(expected[field+'_path']))
        need(row==expected,'Report stage/pure-X/model summary differs: '+stage)
    need(reporter.render(report).encode('utf-8')==(ROOT/report['report_path']).read_bytes(),'Report human/machine content differs')
    proof_path='outputs/round11/backtest_review.json';proof=binding.authority(proof_path);proof_md='reports/round11_backtest_review.md';binding.add(proof_md);binding.mapping(proof['source_sha256'])
    completed=[stage for stage in STAGES if stage in results];evidence={}
    for stage in completed:
        mf=OUT/'models'/stage;ef=OUT/'evaluation'/stage
        paths={'results_sha256':ef/'results.json','model_metadata_sha256':mf/'metadata.json','model_sha256':mf/'model.cbm','feature_state_sha256':mf/'feature_state.json','component_predictions_sha256':mf/'predictions.npy','control_predictions_sha256':ef/(CONTROL+'.npy'),'candidate_predictions_sha256':ef/(CANDIDATE+'.npy')}
        evidence[stage]={key:sha(path) for key,path in paths.items()}
    reporter.validate_backtest_review(proof,plan,accepted,authorities,completed,evidence,sha(ROOT/proof_md))
    expected_review={'status':proof['status'],'candidate_accepted':proof['candidate_accepted'],'checked_stages':completed,'path':proof_path,'sha256':sha(ROOT/proof_path),'report_path':proof_md,'report_sha256':sha(ROOT/proof_md)}
    need(report['independent_backtest_review']==expected_review,'Report independent review differs')
    required=set(binding.values)-{'reports/round11_results.json','reports/round11_results.sha256'}
    required.add('src/round11_report.py')
    need(required<=set(report['source_input_sha256']),'Report source authority closure incomplete')

def collect():
    # Lazy import: only create uses read-only driver validators. No driver entrypoint runs.
    import round11_experiment as driver
    driver.require_no_staging()
    need(not any((OUT/area/'final').exists() for area in ['weights','models','evaluation']) and not (OUT/'release').exists(),'This archiver supports finalized backtests, not final fitting/releases')
    output_paths=[p for p in OUT.rglob('*') if p.is_file()]
    reject_staging_names([p.relative_to(OUT).as_posix() for p in OUT.rglob('*')])
    for area in ['weights','models','evaluation']:
        if (OUT/area).exists():need(all(p.is_dir() and p.name in STAGES for p in (OUT/area).iterdir()),'Unexpected stage entry in '+area)
    need(not any(p.name.lower().startswith('submission') and p.suffix.lower()=='.csv' for p in output_paths),'R11 submission is not a backtest artifact')
    binding=Bindings();plan=driver.plan()
    binding.authority('outputs/round11/candidate_plan.json');domain=binding.authority('outputs/round11/domain_plan.json')
    for field in ['source_sha256','dependency_sha256','final_dependency_sha256']:binding.mapping(plan[field])
    for mapping in plan['stage_dependency_sha256'].values():binding.mapping(mapping)
    for field in ['source_sha256','input_sha256']:binding.mapping(domain[field])
    frozen=binding.authority('outputs/round11/frozen_candidate.json');accepted=binding.authority('outputs/round11/accepted_candidate.json')
    need(frozen['round']==accepted['round']==11 and accepted['candidate']==CANDIDATE,'Wrong acceptance round/candidate')
    for value in [frozen,accepted]:
        need(value['plan_sha256']==sha(OUT/'candidate_plan.json') and value['domain_plan_sha256']==sha(OUT/'domain_plan.json') and value['source_sha256']==plan['source_sha256'],'Final authority source/plan differs')
        binding.mapping(value['source_sha256'])
    need(accepted['frozen_candidate_sha256']==sha(OUT/'frozen_candidate.json'),'Final freeze identity differs')
    for key in ['formulas','thresholds','parameters']:need(frozen[key]==plan[key],'Freeze fixed configuration differs')
    need(accepted['no_reselection_or_parameter_change'] is True and accepted['private_score_known'] is False and accepted['historically_untouched'] is False,'Final policy differs')
    binding.mapping(frozen['development_artifact_sha256']);binding.mapping(accepted['preparation_evidence_sha256'])
    # Expand every old annual geographic manifest's full source/index closure.
    for stage in (*STAGES,'final'):
        oldpath='outputs/round6/geography/'+stage+'/manifest.json';old=binding.authority(oldpath)
        authoritative=plan['final_dependency_sha256'] if stage=='final' else plan['stage_dependency_sha256'][stage]
        need(authoritative.get(oldpath)==sha(ROOT/oldpath),'Old geography manifest absent from price authority')
        for field in ['source_sha256','index_files_sha256']:
            need(all(authoritative.get(k)==v for k,v in old[field].items()),'Old geography closure omitted from price plan');binding.mapping(old[field])
        for item in old['features'].values():
            need(authoritative.get(item['path'])==item['sha256'],'Old feature identity missing from plan');binding.add(item['path'],item['sha256'])
    weights={};results={};models={};late_evidence={};late_manifest_sha={}
    for stage in STAGES:
        wf=OUT/'weights'/stage;mf=OUT/'models'/stage;ef=OUT/'evaluation'/stage
        if wf.exists():
            weight,evidence=driver.verify_weight_manifest(stage,plan,require_pass=False);weights[stage]=weight;binding.mapping(evidence)
            if stage!='development':late_evidence.update(evidence);late_manifest_sha[stage]=sha(wf/'manifest.json')
        need(mf.exists()==ef.exists(),'Incomplete model/evaluation pair for '+stage)
        if ef.exists():
            need(stage in weights and weights[stage]['feasibility_passed'] is True,'Price stage lacks passed weights')
            result=driver.read_result(stage);results[stage]=result;binding.authority(rel(ef/'results.json'))
            for field in ['source_sha256','input_artifact_sha256','output_artifact_sha256']:binding.mapping(result[field])
            expected_outputs={rel(ef/(CONTROL+'.npy')),rel(ef/(CANDIDATE+'.npy'))}
            need(set(result['output_artifact_sha256'])==expected_outputs,'Evaluation prediction inventory differs')
            need({p.name for p in ef.iterdir() if p.is_file()}=={CONTROL+'.npy',CANDIDATE+'.npy','results.json','results.sha256'},'Unexpected evaluation artifacts')
            need(result['target_group_weights']==plan['target_group_weights'] and result['fit_rows']==plan['split_contract'][stage]['fit_rows'] and result['query_rows']==plan['split_contract'][stage]['query_rows'],'Evaluation weights/rows differ')
            for key in ['v3_control_replayed_all_rows','new_model_replayed_all_rows']:need(result[key] is True,'Saved replay proof missing')
            for key in ['training_importance_weights_used_for_scoring','public_score_used','historically_untouched','new_public_score_known','private_score_known']:need(result[key] is False,'Evaluation policy differs')
            models[stage]=check_model(stage,plan,weights[stage],binding,driver)
            need(result['weight_contract']==models[stage]['weight_contract'],'Evaluation training weights differ')
    statuses,failed=fixed_finalization(frozen,accepted,results,weights,set(models))
    need(driver.final_outcome(results,failed)==(accepted['accepted'],accepted['stopping_reason']),'Driver final outcome differs')
    need(accepted['stage_preparation_manifest_sha256']==late_manifest_sha and accepted['preparation_evidence_sha256']==late_evidence,'Final pure-X evidence inventory differs')
    need(accepted['stage_result_sha256']=={s:sha(OUT/'evaluation'/s/'results.json') for s in results},'Final result SHA inventory differs')
    need(frozen['result_sha256']==sha(OUT/'evaluation/development/results.json'),'Freeze development result differs')
    report=binding.authority('reports/round11_results.json');validate_report(report,plan,frozen,accepted,results,weights,models,binding)
    paths=set((ROOT/'src').glob('round11*.py'))|{Path(__file__).resolve()}|set((ROOT/'reports').glob('round11*.md'))|set(output_paths)|{ROOT/'reports/round11_results.json',ROOT/'reports/round11_results.sha256'}
    need(ROOT/'reports/round11_report.md' in paths,'Final report missing')
    for path in sorted(paths):binding.add(rel(path))
    # Every saved JSON receipt is checked; absent receipts are not invented.
    for path in paths:
        if path.suffix=='.sha256':
            target=path.with_suffix('.json');need(target.is_file(),'Receipt has no JSON authority: '+rel(path));locked(target)
    copied={rel(p):{'sha256':binding.values[rel(p)],'bytes':p.stat().st_size} for p in sorted(paths)}
    external={k:{'sha256':v,'bytes':safe(k).stat().st_size} for k,v in sorted(binding.values.items()) if k not in copied}
    state={'round':11,'is_release':False,'accepted_backtest':accepted['accepted'],'selected_candidate':accepted['selected_candidate'],'stopping_reason':accepted['stopping_reason'],'stage_status':statuses,'pure_X_failure_stages':failed,'recommended_version':'v3','recommended_submission':'outputs/submission_v3.csv','new_submission_created':False,'registry_modified':False,'property_labels_or_prediction_values_read_by_archiver':False,'models_refitted_or_row_MAE_recomputed':False,'plan_sha256':sha(OUT/'candidate_plan.json'),'domain_plan_sha256':sha(OUT/'domain_plan.json'),'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),'accepted_candidate_sha256':sha(OUT/'accepted_candidate.json'),'report_json_sha256':sha(ROOT/'reports/round11_results.json'),'historically_untouched':False,'mutable_context_observed_at_creation_not_live_dependencies':dict(binding.mutable_observations)}
    binding.recheck();return copied,external,state

def verify_snapshot():
    manifest=locked(DEST/'snapshot_manifest.json')
    need(manifest['kind']=='round11_finalized_backtest_experiments' and manifest['state']['round']==11 and manifest['state']['is_release'] is False,'Wrong snapshot type')
    need(manifest['state']['recommended_version']=='v3' and manifest['state']['new_submission_created'] is False,'Snapshot claims a release')
    copied=manifest['copied_files'];external=manifest['external_dependencies']
    need(set(copied).isdisjoint(external) and not MUTABLE.intersection(set(copied)|set(external)),'Overlapping/mutable snapshot dependencies')
    need(manifest['copied_file_count']==len(copied) and manifest['external_dependency_count']==len(external),'Snapshot counts differ')
    need(manifest['copied_bytes']==sum(v['bytes'] for v in copied.values()),'Copied bytes differ')
    expected={'snapshot_manifest.json','snapshot_manifest.sha256','SNAPSHOT_README.md'}
    for name,item in copied.items():
        backup=safe('files/'+normalized(name),DEST);need(item['backup_path']==rel(backup),'Snapshot backup path differs')
        need(backup.stat().st_size==item['bytes'] and sha(backup)==item['sha256'],'Backup changed: '+name)
        original=safe(name);need(original.stat().st_size==item['bytes'] and sha(original)==item['sha256'],'Frozen original changed: '+name);expected.add('files/'+name)
    for name,item in external.items():
        path=safe(name);need(path.stat().st_size==item['bytes'] and sha(path)==item['sha256'],'External dependency changed: '+name)
    need((DEST/'SNAPSHOT_README.md').read_bytes()==NOTICE.encode('utf-8') and sha(DEST/'SNAPSHOT_README.md')==manifest['notice_sha256'],'Snapshot notice changed')
    need({p.relative_to(DEST).as_posix() for p in DEST.rglob('*') if p.is_file()}==expected,'Snapshot inventory differs')
    return {'verified':True,'snapshot_sha256':sha(DEST/'snapshot_manifest.json'),'files':len(copied),'external_dependencies':len(external),'copied_bytes':manifest['copied_bytes'],'mutable_entry_points_not_bound':True}

def create():
    need(not DEST.exists(),'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round11_experiments_staging_*')),'Previous archive staging requires inspection; preserve it')
    copied,external,state=collect();staging=DEST.parent/('_round11_experiments_staging_'+uuid.uuid4().hex)
    need(staging.resolve().is_relative_to((ROOT/'runs').resolve()) and not staging.exists(),'Unsafe staging destination');staging.mkdir(parents=True)
    records={}
    for name,item in copied.items():
        source=safe(name);target=safe('files/'+name,staging,exists=False);need(not target.exists(),'Refusing copy overwrite')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
        need(sha(source)==item['sha256']==sha(target) and target.stat().st_size==item['bytes'],'Source/copy changed: '+name)
        records[name]={**item,'backup_path':rel(DEST/'files'/name)}
    for name,item in external.items():need(sha(safe(name))==item['sha256'],'External changed during copy: '+name)
    notice=staging/'SNAPSHOT_README.md';notice.write_bytes(NOTICE.encode('utf-8'))
    manifest={'kind':'round11_finalized_backtest_experiments','created_at_utc':datetime.now(timezone.utc).isoformat(),'state':state,'copied_files':records,'external_dependencies':external,'copied_file_count':len(records),'copied_bytes':sum(x['bytes'] for x in records.values()),'external_dependency_count':len(external),'notice_sha256':sha(notice),'not_self_contained':True,'failed_staging_preserved':True,'registry_and_mutable_status_not_modified_or_bound':True}
    path=staging/'snapshot_manifest.json'
    with path.open('xb') as f:f.write(encoded(manifest))
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')
    need(not DEST.exists(),'Destination concurrently created; staging preserved');staging.rename(DEST)
    result=verify_snapshot();result['status']='created';print(json.dumps(result),flush=True)

def self_check():
    checks=[]
    def reject(name,operation):
        try:operation()
        except (RuntimeError,KeyError):checks.append(name)
        else:raise AssertionError('Expected rejection: '+name)
    b=Bindings();b.declare('src/round11_example.py','a'*64);b.declare('src/round11_example.py','a'*64);checks.append('identical_dependency_merge')
    reject('conflicting_SHA_rejected',lambda:b.declare('src/round11_example.py','b'*64))
    reject('case_alias_rejected',lambda:b.declare('SRC/round11_example.py','a'*64))
    for path in ['../outside','C:/outside','/outside','runs/versions.json','CURRENT_STATUS.md']:
        reject('unsafe_or_mutable_'+path,lambda path=path:b.declare(path,'a'*64))
    for path in ['weights/_development_failed/model','models/.confirmation-failed/model','evaluation/auxiliary/results.json.tmp']:
        reject('staging_'+path,lambda path=path:reject_staging_names([path]))
    reject_staging_names(['weights/development/fold_0/classifier.joblib','research/inventory.json','logs/development.log']);checks.append('successful_inventory_names_allowed')
    def fixture(dev=True,later=True):
        f={'passed':dev,'later_stages_allowed':dev,'selected_candidate':CANDIDATE if dev else CONTROL}
        r={'development':{'decision':{'passed':dev}}};w={'development':{'feasibility_passed':True}}
        if dev and later:
            r.update({s:{'decision':{'passed':True}} for s in STAGES[1:]});w.update({s:{'feasibility_passed':True} for s in STAGES[1:]})
        a={'accepted':dev and later,'selected_candidate':f['selected_candidate'],'stopping_reason':'three_stage_price_gates_passed' if dev and later else 'development_price_gates_failed','stage_status':{s:'price_evaluated' if s in r else 'not_executed_after_development_rejection' for s in STAGES},'pure_X_feasibility_failure_stages':[],'unevaluated_stages':[s for s in STAGES if s not in r],'deployment_decision':'backtest_accepted_pending_final_release' if dev and later else 'keep_v3','new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv'}
        return f,a,r,w
    f,a,r,w=fixture(False);fixed_finalization(f,a,r,w,set(r));checks.append('dev_rejection_without_later_scores')
    f,a,r,w=fixture();fixed_finalization(f,a,r,w,set(r));checks.append('three_passes_remain_backtest_only')
    bad=copy.deepcopy(a);bad['new_submission_created']=True;reject('release_claim_rejected',lambda:fixed_finalization(f,bad,r,w,set(r)))
    bad=copy.deepcopy(r);bad['confirmation']['decision']['passed']=False;reject('inconsistent_acceptance_rejected',lambda:fixed_finalization(f,a,bad,w,set(bad)))
    f,a,r,w=fixture(True,False);reject('missing_later_not_method_rejection',lambda:fixed_finalization(f,a,r,w,set(r)))
    w['confirmation']={'feasibility_passed':False};a.update(accepted=False,stopping_reason='pure_X_feasibility_failed',stage_status={'development':'price_evaluated','confirmation':'pure_X_feasibility_failed_price_not_evaluated','auxiliary':'not_executed_after_pure_X_stop'},pure_X_feasibility_failure_stages=['confirmation'],deployment_decision='keep_v3')
    fixed_finalization(f,a,r,w,set(r));checks.append('verified_pure_X_stop_without_invented_metrics')
    bad=copy.deepcopy(r);bad['confirmation']={'decision':{'passed':True}};reject('price_score_after_failed_weights_rejected',lambda:fixed_finalization(f,a,bad,w,set(bad)))
    reject('partial_unscored_model_rejected',lambda:fixed_finalization(f,a,r,w,{'development','auxiliary'}))
    b.observe_mutable('runs/versions.json','c'*64);need('runs/versions.json' not in b.values and b.mutable_observations['runs/versions.json']=='c'*64,'Mutable observation leaked into dependencies');checks.append('mutable_context_is_observation_not_dependency')
    reject('conflicting_mutable_observation',lambda:b.observe_mutable('runs/versions.json','d'*64))
    f,a,r,w=fixture();r['confirmation']['decision']['passed']=False;a.update(accepted=False,stopping_reason='later_price_gates_failed',deployment_decision='keep_v3');fixed_finalization(f,a,r,w,set(r));checks.append('later_price_failure_keeps_frozen_candidate_but_recommends_v3')
    f,a,r,w=fixture(True,False);w.update(confirmation={'feasibility_passed':False},auxiliary={'feasibility_passed':True});a.update(accepted=False,stopping_reason='pure_X_feasibility_failed',stage_status={'development':'price_evaluated','confirmation':'pure_X_feasibility_failed_price_not_evaluated','auxiliary':'weights_prepared_price_not_executed'},pure_X_feasibility_failure_stages=['confirmation'],deployment_decision='keep_v3');fixed_finalization(f,a,r,w,set(r));checks.append('other_prepared_stage_can_remain_unscored_after_pure_X_stop')
    return {'synthetic_only':True,'no_data_results_models_or_registry_read':True,'files_created':0,'checks':checks,'count':len(checks),'script_sha256':sha(__file__)}

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);group=parser.add_mutually_exclusive_group();group.add_argument('--verify',action='store_true');group.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    elif args.verify:print(json.dumps(verify_snapshot()),flush=True)
    else:create()
