"""Freeze rejected round-four experiments; never publish a release or new CSV.

Run only after final reports/logs/checks are complete. Existing snapshots are
verified, never overwritten. Failed staging is retained; no deletion is used.
"""
from __future__ import annotations
import argparse
import ast
import hashlib
import json
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round4'
DESTINATION = ROOT / 'runs/round4_experiments'
DATA_DIR = ROOT / '台灣房屋銷售價格預測挑戰_回歸/ml-2026-iaii-regression'
RAW_NAMES = {'train':'train-v2.xlsx','valid':'valid-v2.xlsx','test':'test-reindex-test-v2.2.xlsx',
             'sample':'test-reindex-test-sample-submission-v2.csv'}
SUFFIXES = {'.json','.jsonl','.csv','.npy','.pkl','.cbm','.txt','.md','.py','.sha256','.png','.svg','.html','.pdf','.ipynb'}
EXCLUDED_DIRS = {'catboost_info','__pycache__','logs','log','notebook_runtime','matplotlib_cache','cache','data_cache','feature_cache'}
REQUIRED_DOCUMENTS = ['README_round4.md','reports/optimization_round4_report.md','reports/round4_results.json',
    'reports/round4_work_log.md','reports/round4_preprocessing_record.md','reports/round4_final_checks.json']
MANIFEST, RECEIPT, NOTICE_FILE = 'snapshot_manifest.json','snapshot_manifest.sha256','SNAPSHOT_README.md'
NOTICE = '''# 第四輪實驗快照

本資料夾保存第四輪候選、預測、稽核、程式與完整紀錄。四個新候選均未通過
預先固定的開發門檻，選定 v2_control，接受結果為 false；因此維持既有
outputs/submission_v2.csv，沒有產生新提交檔，也不是 v4 release。
本程式不改動 v1、v2、第三輪快照或 runs/versions.json。

此快照不是獨立、免依賴的執行環境。原始資料、大型快取與既有基準／v2 模型
不重複複製；external_dependencies 記錄原始路徑、SHA-256、大小及實際驗證過
的舊備份位置。請保留這些依賴或依原始資料重建快取，不能假設原始資料也有
舊快照備份。程式中的 ROOT 應指向依賴完整的專案根目錄。

copied_files 記錄每個原檔與備份的雜湊和大小。snapshot_manifest.sha256
是 manifest 本身的獨立收據，避免自我雜湊循環。舊年度標籤曾在歷史訓練或
評估中使用，這份快照不宣稱全新、未見標籤；Private 成績仍未知。

暫存、訓練 log 和 live_status 不納入。若封存中斷，暫存目錄留供檢查；不做
遞迴刪除。既有快照與當前內容不一致時一律拒絕覆寫。versions.json 如有複本，
僅代表封存當時的版本上下文，不是新的版本索引。
'''


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def inside(base, path):
    base, path = Path(base).resolve(), Path(path)
    result = (path if path.is_absolute() else base / path).resolve()
    require(result != base and result.is_relative_to(base), 'Path escapes intended root: ' + str(path))
    return result


def rel(path):
    return inside(ROOT, path).relative_to(ROOT.resolve()).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value,ensure_ascii=False,indent=2,sort_keys=True,allow_nan=False)+'\n').encode('utf-8')


def file_record(path):
    path = inside(ROOT,path)
    require(path.is_file(), 'Missing dependency: ' + str(path))
    return {'original_path':rel(path),'original_absolute_path':str(path),'sha256':sha256(path),'bytes':path.stat().st_size}


def verify_record(base, name, digest, size=None):
    path = inside(base,name)
    require(path.is_file(), 'Missing preserved file: ' + str(path))
    require(size is None or path.stat().st_size == size, 'Size differs: ' + str(path))
    require(sha256(path) == digest, 'SHA differs: ' + str(path))


def verify_mapping(mapping):
    for name,digest in mapping.items():
        verify_record(ROOT,name,digest)


def locked_read(path):
    path = inside(ROOT,path)
    receipt = path.with_suffix('.sha256')
    require(receipt.is_file() and receipt.read_text(encoding='ascii').strip() == sha256(path), 'Immutable receipt differs: ' + str(path))
    return read(path)


def verify_contracts():
    names = {'plan':'outputs/round4/candidate_plan.json','freeze':'outputs/round4/frozen_candidate.json',
             'acceptance':'outputs/round4/accepted_candidate.json','development':'outputs/round4/evaluation/development/results.json'}
    contracts = {key:locked_read(ROOT/name) for key,name in names.items()}
    mappings = {'plan':['fit_source_sha256','evaluation_source_sha256','data_sha256'],
                'freeze':['source_sha256','data_sha256','development_artifact_sha256'],
                'acceptance':['evidence_sha256'],
                'development':['source_sha256','data_sha256','input_artifact_sha256','output_artifact_sha256']}
    for key,fields in mappings.items():
        for field in fields:
            verify_mapping(contracts[key][field])
    p,f,a,d = (contracts[key] for key in ['plan','freeze','acceptance','development'])
    plan_sha = sha256(ROOT/names['plan']); freeze_sha = sha256(ROOT/names['freeze'])
    require(all(value['plan_sha256'] == plan_sha for value in [f,a,d]), 'Contracts refer to different plans')
    require(a['frozen_candidate_sha256'] == freeze_sha, 'Acceptance refers to a different freeze')
    require(f['selected_candidate'] == a['selected_candidate'] == d['decision']['selected_candidate'] == 'v2_control', 'This archive requires the retained-v2 outcome')
    require(a['accepted'] is False and a['deployment_decision'] == 'retain_v2', 'Unexpected accepted release; do not archive as rejected experiments')
    require(f['required_components'] == [] and d['decision']['qualified_candidates'] == [], 'Unexpected qualified candidate')
    checks = d['decision']['candidate_checks']
    require(len(checks) == 4 and all(value['eligible'] is False for value in checks.values()), 'Expected four rejected development candidates')
    require(f['candidate_formulas'] == d['candidate_formulas'] == p['candidate_formulas'], 'Formula contract differs')
    require(f['thresholds'] == d['thresholds'] == p['thresholds'], 'Threshold contract differs')
    for folder in [OUT/'models/confirmation',OUT/'models/auxiliary',OUT/'models/final',OUT/'evaluation/confirmation',OUT/'evaluation/auxiliary']:
        require(not folder.exists() or not any(path.is_file() for path in folder.rglob('*')), 'Later-stage artifacts unexpected after control selection')
    require(not list(OUT.rglob('submission*.csv')), 'Unexpected new round-four submission CSV')
    return contracts, {key:{'path':name,'sha256':sha256(ROOT/name),'receipt_sha256':sha256((ROOT/name).with_suffix('.sha256'))}
                       for key,name in names.items()}


def audit_prior_archives():
    """Read-only proof of every v2 and round-three backed-up artifact."""
    previous = ROOT/'runs/round3_experiments'
    r3path = previous/MANIFEST
    r3 = locked_read(r3path)
    require(r3['snapshot_kind'] == 'round3_experiments' and r3['is_release'] is False, 'Wrong round-three snapshot')
    v2path = ROOT/'runs/v2'/MANIFEST
    expected_v2 = [item['sha256'] for item in r3['external_dependencies'] if item['original_path'] == 'runs/v2/snapshot_manifest.json']
    require(len(expected_v2) == 1 and sha256(v2path) == expected_v2[0], 'v2 manifest changed since round-three snapshot')
    v2 = read(v2path)
    require(v2['version'] == 'v2' and v2['files_sha256'], 'Wrong/empty v2 snapshot')
    for name,digest in v2['files_sha256'].items():
        verify_record(ROOT/'runs/v2',name,digest)
    for name,record in r3['copied_files'].items():
        require(record['original_path'] == name and record['backup_path'] == 'runs/round3_experiments/'+name, 'Round-three backup path differs')
        require(record['original_sha256'] == record['backup_sha256'] and record['original_bytes'] == record['backup_bytes'], 'Round-three original/backup proof differs')
        verify_record(previous,name,record['backup_sha256'],record['backup_bytes'])
    expected_files = set(r3['copied_files']) | {MANIFEST,RECEIPT,r3['dependency_notice']}
    require({p.relative_to(previous).as_posix() for p in previous.rglob('*') if p.is_file()} == expected_files, 'Round-three snapshot inventory changed')
    return {'v2':{'manifest_path':rel(v2path),'manifest_sha256':sha256(v2path),'verified_files':len(v2['files_sha256'])},
            'round3_experiments':{'manifest_path':rel(r3path),'manifest_sha256':sha256(r3path),
                'receipt_sha256':sha256(previous/RECEIPT),'notice_sha256':sha256(previous/r3['dependency_notice']),
                'verified_files':len(r3['copied_files'])}}


def local_sources(paths):
    pending, found = list(paths), set()
    while pending:
        path = inside(ROOT,pending.pop())
        if path in found:
            continue
        require(path.is_file(), 'Missing source file: ' + str(path)); found.add(path)
        for node in ast.walk(ast.parse(path.read_text(encoding='utf-8-sig'))):
            modules = [item.name for item in node.names] if isinstance(node,ast.Import) else [node.module] if isinstance(node,ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = ROOT/'src'/(module.split('.')[0]+'.py')
                if candidate.is_file() and candidate.resolve() not in found:
                    pending.append(candidate)
    return found


def collect_files(contracts):
    files, omitted = set(), []
    for folder,directories,names in os.walk(OUT,followlinks=False):
        parent = Path(folder); keep = []
        for name in directories:
            path = parent/name
            if name in EXCLUDED_DIRS or name.startswith('.') or name.startswith('live_status') or path.is_symlink():
                omitted.append({'path':str(path.relative_to(ROOT)),'reason':'temporary/runtime/log/cache/live-status directory or symbolic link'})
            else:
                keep.append(name)
        directories[:] = keep
        for name in names:
            path = parent/name
            if path.is_symlink() or name.startswith(('.', 'live_status')) or path.suffix.lower() in {'.tmp','.temp','.lock','.log','.pyc'}:
                omitted.append({'path':str(path.relative_to(ROOT)),'reason':'temporary/log/live status or symbolic link'})
            elif path.suffix.lower() in SUFFIXES:
                files.add(inside(ROOT,path))
            else:
                omitted.append({'path':str(path.relative_to(ROOT)),'reason':'outside documented artifact types'})
    for name in REQUIRED_DOCUMENTS:
        path = inside(ROOT,name); require(path.is_file(), 'Finish required document before archiving: '+name); files.add(path)
    files.update(path.resolve() for path in (ROOT/'reports').glob('round4*') if path.is_file())
    sources = set((ROOT/'src').glob('*round4*.py')) | {Path(__file__).resolve()}
    sources.update(ROOT/name for name in contracts['freeze']['source_sha256'] if name.startswith('src/') and name.endswith('.py'))
    files.update(local_sources(sources))
    requirements = list(ROOT.glob('requirements*.txt'))
    require(requirements, 'Missing requirement files'); files.update(path.resolve() for path in requirements)
    for name in ['runs/versions.json','reports/kaggle_submission_history.json','solution_round4.ipynb']:
        if (ROOT/name).is_file():
            files.add((ROOT/name).resolve())
    for path in files:
        name = rel(path)
        require(not name.startswith('runs/') or name == 'runs/versions.json', 'Never recursively copy older snapshots')
        require(not name.startswith('data_cache/') and not path.is_relative_to(DATA_DIR.resolve()), 'Raw/cache dependency must not be copied')
        require(not name.startswith('outputs/round3/baseline/'), 'Old baseline models remain external')
    return sorted(files), sorted(omitted,key=lambda item:item['path'])


def external_dependencies(copy_files,contracts):
    copied, references = {rel(path) for path in copy_files}, {}
    def add(path,reason,expected=None):
        path = inside(ROOT,path); require(path.is_file(), 'Missing external dependency: '+str(path))
        name = rel(path)
        if expected is not None:
            verify_record(ROOT,name,expected)
        if name in copied:
            return
        item = references.setdefault(name,{'reasons':set(),'recorded_sha256':set()}); item['reasons'].add(reason)
        if expected is not None:
            item['recorded_sha256'].add(expected)
    mapping_fields = {'source_sha256','fit_source_sha256','evaluation_source_sha256','data_sha256','input_artifact_sha256',
        'output_artifact_sha256','development_artifact_sha256','evidence_sha256','dependency_script_sha256','inference_source_sha256'}
    path_fields = {'model_path':'model_sha256','reference_path':'reference_sha256','category_maps_path':'category_maps_sha256',
                   'known_projects_path':'known_projects_sha256','pickle_file':'pickle_sha256','metadata_path':None}
    def walk(value,context):
        if isinstance(value,list):
            for child in value:
                walk(child,context)
        elif isinstance(value,dict):
            for field,hash_field in path_fields.items():
                if isinstance(value.get(field),str):
                    add(value[field],context,value.get(hash_field) if hash_field else None)
            for key,child in value.items():
                if key in mapping_fields and isinstance(child,dict):
                    for name,digest in child.items():
                        if isinstance(digest,str) and len(digest) == 64:
                            if name in RAW_NAMES:
                                add(DATA_DIR/RAW_NAMES[name],context,digest)
                            elif (ROOT/name).is_file():
                                add(name,context,digest)
                walk(child,context)
    inputs = read(ROOT/'reports/run_manifest.json')['input_sha256']
    require(set(inputs) == set(RAW_NAMES.values()), 'Expected four original supplied files')
    for name,digest in inputs.items():
        add(DATA_DIR/name,'original supplied input; not copied',digest)
    for path in (ROOT/'data_cache').iterdir():
        if path.is_file() and path.suffix in {'.pkl','.json','.npy','.parquet'}:
            add(path,'data/feature cache; retain or rebuild, not copied')
    for name,value in contracts.items():
        walk(value,'round-four '+name+' contract')
    for path in copy_files:
        if path.suffix == '.json':
            walk(read(path),rel(path))
    for name in ['runs/v1/models/final_config.json','models/round2/release_config.json']:
        add(name,'retained previous release configuration'); walk(read(ROOT/name),name)
    for name in ['runs/v2/snapshot_manifest.json','runs/round3_experiments/snapshot_manifest.json','runs/round3_experiments/snapshot_manifest.sha256']:
        add(name,'verified previous snapshot proof')
    v2 = read(ROOT/'runs/v2'/MANIFEST)
    submission_sha = v2['files_sha256']['outputs/submission_v2.csv']
    add('outputs/submission_v2.csv','retained recommended v2 submission',submission_sha)
    result = []
    for name,item in sorted(references.items()):
        record = file_record(name); expected = sorted(item['recorded_sha256'])
        require(all(digest == record['sha256'] for digest in expected), 'Inconsistent dependency hashes: '+name)
        backups = []
        for prior in ['v1','v2','round3_experiments']:
            candidate = ROOT/'runs'/prior/name
            if candidate.is_file() and sha256(candidate) == record['sha256']:
                backups.append(rel(candidate))
        if name.startswith(('runs/v1/','runs/v2/','runs/round3_experiments/')):
            backups.append(name)
        record.update({'reasons':sorted(item['reasons']),'recorded_sha256':expected,'recorded_hashes_match_current':True,'existing_backups':sorted(set(backups))})
        result.append(record)
    return result


def make_plan():
    contracts,proof = verify_contracts()
    prior = audit_prior_archives()
    versions = file_record('runs/versions.json')
    require(read(ROOT/'runs/versions.json')['recommended_version'] == 'v2', 'Expected retained v2 recommendation')
    files,omitted = collect_files(contracts)
    return {'originals':{rel(path):file_record(path) for path in files},'external_dependencies':external_dependencies(files,contracts),
            'contract_receipts_verified':proof,'prior_archives_verified':prior,'versions_context':versions,'omitted_paths':omitted,
            'selected_candidate':'v2_control','accepted':False,'recommended_submission':'outputs/submission_v2.csv','new_release':False}


def check_sources(plan):
    for name,record in plan['originals'].items():
        verify_record(ROOT,name,record['sha256'],record['bytes'])
    for record in plan['external_dependencies']:
        verify_record(ROOT,record['original_path'],record['sha256'],record['bytes'])
    require(audit_prior_archives() == plan['prior_archives_verified'], 'Prior archive changed during snapshot creation')
    _,proof = verify_contracts(); require(proof == plan['contract_receipts_verified'], 'Round-four contracts changed')
    record = plan['versions_context']; verify_record(ROOT,record['original_path'],record['sha256'],record['bytes'])


def verify_snapshot(folder,plan):
    path = folder/MANIFEST; receipt = folder/RECEIPT
    require(path.is_file() and receipt.is_file() and receipt.read_text(encoding='ascii').strip() == sha256(path), 'Snapshot manifest receipt missing/changed')
    manifest = read(path)
    require(manifest['snapshot_kind'] == 'round4_experiments' and manifest['is_release'] is False, 'Wrong snapshot kind')
    require(manifest['original_plan'] == plan, 'Current sources/dependencies differ; refusing snapshot overwrite')
    expected = set(plan['originals']) | {MANIFEST,RECEIPT,NOTICE_FILE}
    require({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == expected, 'Snapshot inventory differs')
    require(set(manifest['copied_files']) == set(plan['originals']), 'Snapshot catalog differs')
    for name,record in manifest['copied_files'].items():
        original = plan['originals'][name]
        require(record['original_sha256'] == original['sha256'] == record['backup_sha256'], 'Original/backup hash differs: '+name)
        require(record['original_bytes'] == original['bytes'] == record['backup_bytes'], 'Original/backup size differs: '+name)
        require(record['backup_path'] == 'runs/round4_experiments/'+name, 'Backup path differs')
        verify_record(folder,name,record['backup_sha256'],record['backup_bytes'])
    require((folder/NOTICE_FILE).read_bytes() == NOTICE.encode('utf-8'), 'Dependency notice differs')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--verify-only',action='store_true')
    args = parser.parse_args(); plan = make_plan()
    destination, runs = inside(ROOT,DESTINATION), inside(ROOT,'runs')
    require(destination.parent == runs and destination.name == 'round4_experiments', 'Unsafe snapshot destination')
    if destination.exists():
        manifest = verify_snapshot(destination,plan); check_sources(plan)
        print(json.dumps({'status':'verified_existing','snapshot':str(destination),'files':len(manifest['copied_files'])}),flush=True)
        return
    require(not args.verify_only, 'No round-four snapshot exists yet')
    runs.mkdir(parents=True,exist_ok=True)
    copied_bytes = sum(record['bytes'] for record in plan['originals'].values())
    require(shutil.disk_usage(runs).free > copied_bytes+16*1024*1024, 'Insufficient snapshot disk space')
    stage = inside(runs,'.round4-experiments-staging-'+uuid.uuid4().hex)
    require(stage.parent == runs, 'Unsafe staging directory'); stage.mkdir()
    try:
        copied = {}
        for name,original in sorted(plan['originals'].items()):
            source,backup = inside(ROOT,name),inside(stage,name); backup.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,backup); digest,size = sha256(backup),backup.stat().st_size
            require(digest == original['sha256'] and size == original['bytes'], 'Copy verification failed: '+name)
            copied[name] = {'original_path':name,'original_absolute_path':str(source),'original_sha256':digest,'original_bytes':size,
                            'backup_path':'runs/round4_experiments/'+name,'backup_sha256':digest,'backup_bytes':size}
        manifest = {'format_version':1,'snapshot_kind':'round4_experiments','created_at_utc':datetime.now(timezone.utc).isoformat(),
            'is_release':False,'release_version':None,'self_contained':False,'dependency_notice':NOTICE_FILE,
            'recommended_submission':'outputs/submission_v2.csv','new_submission_created':False,'selected_candidate':'v2_control','accepted':False,
            'copied_files':copied,'external_dependencies':plan['external_dependencies'],'original_plan':plan,
            'prior_archives_modified':False,'versions_modified':False,'raw_data_and_large_caches_duplicated':False,
            'prior_baseline_models_duplicated':False,'manifest_self_excluded':True,'manifest_sha256_receipt':RECEIPT}
        (stage/NOTICE_FILE).write_bytes(NOTICE.encode('utf-8')); (stage/MANIFEST).write_bytes(encoded(manifest))
        (stage/RECEIPT).write_text(sha256(stage/MANIFEST)+'\n',encoding='ascii')
        verify_snapshot(stage,plan); check_sources(plan)
        require(not destination.exists() and stage.resolve().parent == runs and destination.resolve().parent == runs, 'Unsafe move or destination appeared')
        stage.rename(destination)
    except BaseException:
        print('Archive not finalized; staging retained for inspection: '+str(stage),flush=True)
        raise
    verify_snapshot(destination,plan)
    print(json.dumps({'status':'created','snapshot':str(destination),'files':len(copied),'copied_bytes':copied_bytes,
        'external_dependencies':len(plan['external_dependencies']),'snapshot_manifest_sha256':sha256(destination/MANIFEST),
        'is_release':False,'self_contained':False,'prior_archives_unchanged':True,'versions_unchanged':True}),flush=True)


if __name__ == '__main__':
    main()
