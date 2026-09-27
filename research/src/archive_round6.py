"""Archive a rejected round-six experiment; never create a release or submission.

DRAFT: execute only after accepted=false, final reports and checks exist.
Accepted candidates are explicitly rejected here and need a separate release
workflow. --verify checks an existing snapshot without overwriting anything.
Only named authoritative contracts are followed for dependencies; historical
QA JSON is preserved as evidence and is not rebound to today's sources.
"""
from __future__ import annotations
import argparse,ast,os,shutil,uuid
from datetime import datetime,timezone
from pathlib import Path
from archive_round5 import (ROOT,require,sha256,inside,rel,read,encoded,
                            file_record,verify_record,local_sources)

OUT=ROOT/'outputs/round6'
DESTINATION=ROOT/'runs/round6_experiments'
MANIFEST='snapshot_manifest.json'
RECEIPT='snapshot_manifest.sha256'
NOTICE_FILE='SNAPSHOT_README.md'
V2_SHA='ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
R5_SHA='10444c2e1be03f2e3a6354784c4b6cb9bf5baf8dcf2b5b4e99026c5c9f75d4bf'
HISTORICAL_DRAFT='outputs/round6/audit/independent_preflight_draft.json'
OLD_PREPARE_SHA='75cea3e93039a7b2a52e28ed787d43eb735930865b80bdf74eac5421dd2787c7'
SUFFIXES={'.json','.jsonl','.csv','.npy','.pkl','.cbm','.txt','.md','.py','.sha256','.png','.svg','.html','.pdf','.ipynb'}
OMIT_DIRS={'catboost_info','__pycache__','logs','log','notebook_runtime','matplotlib_cache'}
REQUIRED_DOCUMENTS=['README_round6.md','reports/optimization_round6_report.md','reports/round6_results.json',
 'reports/round6_work_log.md','reports/round6_preprocessing_record.md','reports/round6_final_checks.json']
NOTICE='''# 第六輪未採用實驗快照

本快照只接受 accepted=false 的第六輪結果，保存本輪模型、預測、8 欄地理特徵、
來源程式、固定計畫、收據與最終紀錄。維持 outputs/submission_v2.csv，沒有新增
release 或提交檔，也不改 versions.json、v2 或第五輪等既有快照。

這不是獨立免依賴的執行環境。大型 PBF／道路／行政面、地理原始匹配、原始資料、
大型快取與既有基準模型均只列於 external_dependencies；保留原路徑、SHA、大小與
實際驗證過的舊備份位置。只複製權威合約明列的必要小型地理 audit／receipt，
不遞迴複製整個 geography 或 runs。重播需保留或重建所有外部依賴。

independent_preflight_draft.json 是修正前的歷史工程證據，其中舊 prepare source
未另存副本；不把舊 SHA 拿去驗目前 source，也不宣稱該歷史草稿可完整重播。
正式 fit／evaluate source 以凍結 plan、final preflight、prepared manifest 與
評估合約為準。其他歷史 audit 原樣保存，不盲目追索其中每個舊 source SHA。

copied_files 記錄每份原檔和備份的 SHA 與大小；manifest 自己另有 SHA 收據，
避免自我雜湊循環。失敗 staging 保留，不遞迴刪除；既有快照不得覆写。
過往年度標籤已被歷史訓練／評估使用，這不代表全新未見標籤或保證 Private 成績。
'''


def locked(path):
    path=inside(ROOT,path);receipt=path.with_suffix('.sha256')
    require(path.is_file() and receipt.is_file(),'Missing immutable contract: '+str(path))
    require(receipt.read_text(encoding='ascii').strip()==sha256(path),'Contract receipt differs: '+str(path))
    return read(path)


def dependency_registry():
    expected={};reasons={}
    def add(name,digest=None,reason='explicit dependency'):
        path=inside(ROOT,name);key=rel(path)
        require(path.is_file(),'Missing dependency: '+key)
        actual=sha256(path)
        require(digest is None or digest==actual,'Recorded dependency SHA differs: '+key)
        require(key not in expected or expected[key]==actual,'Conflicting dependency declaration: '+key)
        expected[key]=actual;reasons.setdefault(key,set()).add(reason)
        return actual
    def mapping(value,reason):
        require(isinstance(value,dict),'Expected dependency map: '+reason)
        for name,digest in value.items():
            require(isinstance(digest,str) and len(digest)==64,'Malformed dependency SHA: '+str(name))
            add(name,digest,reason)
    return expected,reasons,add,mapping


def verify_prior(plan,add):
    path=ROOT/'runs/round5_experiments'/MANIFEST
    require(plan['prior_round5_manifest_sha256']==R5_SHA,'Unexpected prior R5 manifest')
    add(path,R5_SHA,'unchanged fifth-round snapshot')
    old=locked(path);folder=path.parent
    require(old['snapshot_kind']=='round5_experiments' and old['is_release'] is False,'Wrong prior snapshot')
    for name,record in old['copied_files'].items():
        require(record['original_path']==name and record['backup_path']=='runs/round5_experiments/'+name,'Prior backup path differs')
        require(record['original_sha256']==record['backup_sha256'] and record['original_bytes']==record['backup_bytes'],'Prior backup proof differs')
        verify_record(folder,name,record['backup_sha256'],record['backup_bytes'])
    inventory={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(inventory==set(old['copied_files'])|{MANIFEST,RECEIPT,old['dependency_notice']},'Prior R5 inventory differs')
    source=folder/'src/archive_round5.py'
    tree=ast.parse(source.read_text(encoding='utf-8-sig'))
    notices=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
             and any(isinstance(target,ast.Name) and target.id=='NOTICE' for target in node.targets)]
    require(len(notices)==1 and (folder/old['dependency_notice']).read_bytes()==notices[0].encode('utf-8'),'Prior R5 notice differs')
    add(folder/RECEIPT,reason='prior R5 receipt');add(folder/old['dependency_notice'],reason='prior R5 dependency notice')
    prior_v2=old['original_plan']['prior_archives_verified']['v2']
    add(prior_v2['manifest_path'],prior_v2['manifest_sha256'],'v2 snapshot retained since R5')
    v2=read(ROOT/prior_v2['manifest_path']);require(v2['version']=='v2','Wrong retained release')
    for name,digest in v2['files_sha256'].items():verify_record(ROOT/'runs/v2',name,digest)
    require(v2['files_sha256']['outputs/submission_v2.csv']==V2_SHA,'Retained submission SHA differs')
    add('outputs/submission_v2.csv',V2_SHA,'retained recommended submission; no new CSV')
    versions=read(ROOT/'runs/versions.json');require(versions['recommended_version']=='v2','Recommendation unexpectedly changed')
    add('runs/versions.json',reason='unchanged version-index context')
    add('reports/run_manifest.json',reason='original four-input inventory')
    inputs=read(ROOT/'reports/run_manifest.json')['input_sha256']
    require(set(inputs)=={'train-v2.xlsx','valid-v2.xlsx','test-reindex-test-v2.2.xlsx','test-reindex-test-sample-submission-v2.csv'},'Original input inventory differs')
    for filename,digest in inputs.items():
        matches=[item for item in old['external_dependencies'] if Path(item['original_path']).name==filename and item['sha256']==digest]
        require(len(matches)==1,'Original input path ambiguous or absent from preserved R5 inventory: '+filename)
        add(matches[0]['original_path'],digest,'original supplied input; not copied')
    label_names={'train':'train-v2.xlsx','valid':'valid-v2.xlsx','test':'test-reindex-test-v2.2.xlsx'}
    require(plan['original_input_sha256']=={k:inputs[v] for k,v in label_names.items()},'R6 original inputs differ')
    return {'round5_manifest_sha256':R5_SHA,'round5_verified_files':len(old['copied_files']),
            'v2_manifest_path':prior_v2['manifest_path'],'v2_manifest_sha256':prior_v2['manifest_sha256'],
            'v2_verified_files':len(v2['files_sha256']),'recommended_submission_sha256':V2_SHA}


def authoritative_contracts(add,mapping):
    names={'plan':OUT/'candidate_plan.json','freeze':OUT/'frozen_candidate.json','acceptance':OUT/'accepted_candidate.json'}
    # Reject accepted releases before inspecting or collecting experiment artifacts.
    acceptance=locked(names['acceptance'])
    require(acceptance['accepted'] is False,'Accepted candidate: this rejected-experiment archiver refuses; use the formal release workflow')
    require(acceptance['deployment_decision']=='retain_v2' and acceptance['candidate_reselection_performed'] is False,'Expected retained-v2 outcome')
    contracts={key:locked(path) for key,path in names.items()}
    plan,frozen=contracts['plan'],contracts['freeze'];selected=frozen['selected_candidate']
    require(plan['round']==6 and plan['control']=='v2_control','Wrong round/control')
    require(set(plan['candidates'])=={'geography_blend'} and set(plan['components'])=={'geography_catboost'},'Unexpected planned candidate/component')
    require(selected in ('v2_control','geography_blend') and acceptance['selected_candidate']==selected,'Selection differs')
    plan_sha=sha256(names['plan']);freeze_sha=sha256(names['freeze'])
    require(frozen['plan_sha256']==acceptance['plan_sha256']==plan_sha and acceptance['frozen_candidate_sha256']==freeze_sha,'Plan/freeze identity differs')
    require(frozen['required_components']==plan['required_components'][selected],'Frozen component requirement differs')
    for field in ['candidate_formulas','thresholds']:
        require(frozen[field]==plan[field],'Frozen '+field+' differs')
    for field in ['fit_source_sha256','evaluation_source_sha256','data_sha256','research_evidence_sha256','geography_transform_source_sha256']:
        mapping(plan[field],'frozen plan '+field)
    for field in ['source_sha256','data_sha256','development_artifact_sha256']:mapping(frozen[field],'frozen selection '+field)
    mapping(acceptance['evidence_sha256'],'final rejection evidence')
    stages=['development'] if selected=='v2_control' else ['development','confirmation','auxiliary']
    for stage in stages:
        path=OUT/'evaluation'/stage/'results.json';value=locked(path)
        names[stage]=path;contracts[stage]=value
        require(value['stage']==stage and value['plan_sha256']==plan_sha,'Evaluation stage/plan differs')
        require(set(value['evaluated_candidates'])=={'v2_control','geography_blend'},'Unexpected evaluated candidate list')
        require(value['candidate_formulas']==plan['candidate_formulas'] and value['thresholds']==plan['thresholds'],'Evaluation rules differ')
        for field in ['source_sha256','data_sha256','input_artifact_sha256','output_artifact_sha256']:mapping(value[field],stage+' '+field)
        if stage=='development':require(value['decision']['selected_candidate']==selected,'Development selection differs')
        else:require(value['decision']==acceptance['stage_decisions'][stage],'Acceptance stage decision differs')
    if selected=='v2_control':
        require(acceptance['stage_decisions']=={},'Unexpected later-stage decisions after development rejection')
        for family in ['models','evaluation','baseline']:
            for stage in ['confirmation','auxiliary','final']:
                folder=OUT/family/stage
                require(not folder.exists() or not any(p.is_file() for p in folder.rglob('*')),'Unexpected later-stage artifact after development rejection')
    else:
        require(set(acceptance['stage_decisions'])=={'confirmation','auxiliary'} and not all(v['passed'] for v in acceptance['stage_decisions'].values()),'Rejection is inconsistent with later-stage decisions')
    require(not list(OUT.rglob('submission*.csv')),'A rejected experiment must not contain a new submission')
    require(not (OUT/'models/final').exists(),'Final trained model is outside this rejected-experiment archive scope')
    for key,path in names.items():
        add(path,reason='authoritative '+key+' contract');add(path.with_suffix('.sha256'),reason='authoritative '+key+' receipt')
    qa=read(OUT/'audit/preflight_final.json');require(qa['status']=='passed','Final synthetic preflight failed/missing')
    for field,name in [('helper_source_sha256','src/round6_geo.py'),('evaluator_source_sha256','src/round6_evaluate.py'),('preflight_source_sha256','src/round6_preflight.py')]:add(name,qa[field],'final preflight source')
    # Prepared manifests are current authority, unlike historical draft QA.
    for path in sorted((OUT/'geography').glob('*/manifest.json')):
        geo=locked(path);require(geo['geography_provenance_chain']['verified'] is True,'Prepared geography chain not verified')
        require(geo['stage']==path.parent.name,'Geography stage differs')
        mapping(geo['source_sha256'],'prepared '+geo['stage']+' source/input chain')
        mapping(geo['index_files_sha256'],'prepared '+geo['stage']+' index files')
        for item in geo['features'].values():add(item['path'],item['sha256'],'prepared encoded geography')
        add(path,reason='prepared geographic manifest');add(path.with_suffix('.sha256'),reason='prepared geography receipt')
    return contracts,{key:{'path':rel(path),'sha256':sha256(path),'receipt_sha256':sha256(path.with_suffix('.sha256'))} for key,path in names.items()},stages


def collect_files(expected):
    files=set();omitted=[]
    for folder,directories,names in os.walk(OUT,followlinks=False):
        parent=Path(folder);keep=[]
        for name in directories:
            path=parent/name
            if name in OMIT_DIRS or name.startswith(('.', 'live_status')) or path.is_symlink():omitted.append(rel(path))
            else:keep.append(name)
        directories[:]=keep
        for name in names:
            path=parent/name
            if path.is_symlink() or name.startswith(('.', 'live_status')) or path.suffix.lower() in {'.tmp','.temp','.lock','.log','.pyc'}:omitted.append(rel(path))
            elif path.suffix.lower() in SUFFIXES:files.add(path.resolve())
            else:omitted.append(rel(path))
    for name in REQUIRED_DOCUMENTS:
        path=inside(ROOT,name);require(path.is_file(),'Finish required report before archive: '+name);files.add(path)
    files.update(p.resolve() for p in (ROOT/'reports').glob('*round6*') if p.is_file())
    sources=set((ROOT/'src').glob('*round6*.py'))|{Path(__file__).resolve()}
    for name in expected:
        path=ROOT/name
        if name.startswith('src/') and path.suffix=='.py':sources.add(path)
        elif name.startswith('reports/') and path.suffix in {'.json','.md','.png','.svg','.txt','.pdf'}:files.add(path.resolve())
        elif name.startswith(('outputs/geography/','data_external/geography/')) and path.suffix in {'.json','.sha256','.md','.txt'}:
            files.add(path.resolve())  # exact declared compact evidence, never the whole directory
    files.update(local_sources(sources));files.update(p.resolve() for p in ROOT.glob('requirements*.txt'))
    for name in ['reports/kaggle_submission_history.json','data_external/geography/ATTRIBUTION.md','solution_round6.ipynb']:
        if (ROOT/name).is_file():files.add((ROOT/name).resolve())
    for path in files:
        name=rel(path)
        require(not name.startswith(('runs/','data_cache/')) and path.suffix.lower() not in {'.xlsx','.pbf'},'Large/raw/old snapshot dependency must remain external')
        require(not name.startswith(('outputs/round3/baseline/','outputs/round2/')),'Old model dependency must remain external')
        require(not (name.startswith(('outputs/geography/','data_external/geography/')) and path.suffix in {'.pkl','.jsonl','.npy','.cbm'}),'Large geography must remain external')
    return sorted(files),sorted(omitted)


def build_plan():
    expected,reasons,add,mapping=dependency_registry()
    contracts,proof,stages=authoritative_contracts(add,mapping)
    prior=verify_prior(contracts['plan'],add)
    files,omitted=collect_files(expected)
    copied={rel(path) for path in files}
    external=[]
    for name,digest in sorted(expected.items()):
        if name in copied:continue
        record=file_record(name);require(record['sha256']==digest,'External dependency changed')
        backups=[]
        for kind in ['v1','v2','round3_experiments','round4_experiments','round5_experiments']:
            candidate=ROOT/'runs'/kind/name
            if candidate.is_file() and sha256(candidate)==digest:backups.append(rel(candidate))
        if name.startswith('runs/'):backups.append(name)
        record.update({'reasons':sorted(reasons[name]),'recorded_sha256':digest,'existing_backups':sorted(set(backups))})
        external.append(record)
    historical=[]
    path=ROOT/HISTORICAL_DRAFT
    if path.is_file():
        historical.append({'audit_path':HISTORICAL_DRAFT,'audit_sha256':sha256(path),
          'historical_original_path':'src/round6_prepare_geo.py','recorded_historical_sha256':OLD_PREPARE_SHA,
          'historical_source_copy_preserved':False,'replay_certified':False,
          'interpretation':'Historical draft before provenance-chain fix; not current fitting/evaluation authority. No retroactive source copy or receipt is claimed.'})
    return {'originals':{rel(p):file_record(p) for p in files},'external_dependencies':external,
            'contract_receipts_verified':proof,'prior_archives_verified':prior,'historical_evidence':historical,
            'dependency_collection_policy':'Only current plan/freeze/acceptance/evaluation/prepared manifests; no recursive historical QA JSON rebinding',
            'omitted_paths':omitted,'accepted':False,'selected_candidate':contracts['acceptance']['selected_candidate'],
            'evaluated_stages':stages,'recommended_submission':'outputs/submission_v2.csv','recommended_submission_sha256':V2_SHA,'new_release':False}


def check_current(plan):
    for name,record in plan['originals'].items():verify_record(ROOT,name,record['sha256'],record['bytes'])
    for record in plan['external_dependencies']:verify_record(ROOT,record['original_path'],record['sha256'],record['bytes'])
    require(build_plan()==plan,'Current authoritative contracts or exact artifact inventory changed')


def verify_snapshot(folder,plan):
    folder=Path(folder).resolve();path=folder/MANIFEST;receipt=folder/RECEIPT
    require(path.is_file() and receipt.is_file() and receipt.read_text(encoding='ascii').strip()==sha256(path),'Snapshot manifest receipt differs')
    manifest=read(path)
    require(manifest['snapshot_kind']=='round6_experiments' and manifest['is_release'] is False and manifest['accepted'] is False,'Wrong snapshot type/outcome')
    require(manifest['original_plan']==plan,'Snapshot input plan differs')
    require(set(manifest['copied_files'])==set(plan['originals']),'Snapshot file catalog differs')
    actual={p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    require(actual==set(plan['originals'])|{MANIFEST,RECEIPT,NOTICE_FILE},'Snapshot file inventory differs')
    for name,item in manifest['copied_files'].items():
        original=plan['originals'][name]
        require(item['original_path']==name and item['backup_path']=='runs/round6_experiments/'+name,'Snapshot record path differs')
        require(item['original_sha256']==item['backup_sha256']==original['sha256'] and item['original_bytes']==item['backup_bytes']==original['bytes'],'Snapshot original/copy proof differs')
        verify_record(folder,name,item['backup_sha256'],item['backup_bytes'])
    require((folder/NOTICE_FILE).read_bytes()==NOTICE.encode('utf-8'),'Snapshot notice differs')
    return manifest


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--verify','--verify-only',dest='verify',action='store_true')
    args=parser.parse_args();destination=inside(ROOT,DESTINATION);runs=inside(ROOT,'runs')
    require(destination.parent==runs and destination.name=='round6_experiments','Unsafe snapshot destination')
    if args.verify:
        require(destination.is_dir(),'No round-six experiment snapshot to verify')
        plan=read(destination/MANIFEST)['original_plan'];verify_snapshot(destination,plan);check_current(plan)
        print(encoded({'status':'verified','files':len(plan['originals']),'manifest_sha256':sha256(destination/MANIFEST)}).decode(),flush=True);return
    require(not destination.exists(),'Snapshot already exists; use --verify. Never overwrite runs/round6_experiments')
    plan=build_plan();runs.mkdir(parents=True,exist_ok=True)
    copied_bytes=sum(r['bytes'] for r in plan['originals'].values())
    require(shutil.disk_usage(runs).free>copied_bytes+16*1024*1024,'Insufficient snapshot disk space')
    staging=inside(runs,'.round6-experiments-staging-'+uuid.uuid4().hex)
    require(staging.parent==runs,'Unsafe staging directory');staging.mkdir()
    try:
        copied={}
        for name,record in sorted(plan['originals'].items()):
            source=inside(ROOT,name);backup=inside(staging,name);backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,backup);verify_record(staging,name,record['sha256'],record['bytes'])
            copied[name]={'original_path':name,'original_absolute_path':str(source),'original_sha256':record['sha256'],'original_bytes':record['bytes'],
                          'backup_path':'runs/round6_experiments/'+name,'backup_sha256':sha256(backup),'backup_bytes':backup.stat().st_size}
        manifest={'format_version':1,'snapshot_kind':'round6_experiments','created_at_utc':datetime.now(timezone.utc).isoformat(),
          'is_release':False,'release_version':None,'self_contained':False,'dependency_notice':NOTICE_FILE,
          'accepted':False,'selected_candidate':plan['selected_candidate'],'recommended_submission':'outputs/submission_v2.csv',
          'new_submission_created':False,'copied_files':copied,'external_dependencies':plan['external_dependencies'],
          'original_plan':plan,'prior_archives_modified':False,'versions_modified':False,
          'raw_data_and_large_geography_duplicated':False,'old_baseline_models_duplicated':False,
          'manifest_self_excluded':True,'manifest_sha256_receipt':RECEIPT}
        (staging/NOTICE_FILE).write_bytes(NOTICE.encode('utf-8'));(staging/MANIFEST).write_bytes(encoded(manifest))
        (staging/RECEIPT).write_text(sha256(staging/MANIFEST)+'\n',encoding='ascii')
        verify_snapshot(staging,plan);check_current(plan)
        require(not destination.exists() and staging.parent==runs and destination.parent==runs,'Unsafe publication path or destination appeared')
        staging.rename(destination)
    except BaseException:
        print('Archive not finalized; staging retained: '+str(staging),flush=True);raise
    verify_snapshot(destination,plan)
    print(encoded({'status':'created','snapshot':rel(destination),'files':len(copied),'copied_bytes':copied_bytes,
                   'external_dependencies':len(plan['external_dependencies']),'manifest_sha256':sha256(destination/MANIFEST),'is_release':False}).decode(),flush=True)


if __name__=='__main__':main()
