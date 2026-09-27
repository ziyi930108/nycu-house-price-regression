"""Freeze a round-three experiment snapshot; this does not publish a v3 release.

Run after experiment reports are complete. Existing snapshots are verified,
never overwritten. No recursive deletion is performed, including on failure.
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
DESTINATION = ROOT / 'runs' / 'round3_experiments'
DATA_DIR = ROOT / '台灣房屋銷售價格預測挑戰_回歸' / 'ml-2026-iaii-regression'
RAW_NAMES = {'train': 'train-v2.xlsx', 'valid': 'valid-v2.xlsx',
             'test': 'test-reindex-test-v2.2.xlsx',
             'sample': 'test-reindex-test-sample-submission-v2.csv'}
CORE_SUFFIXES = {'.json', '.jsonl', '.csv', '.npy', '.pkl', '.cbm', '.txt', '.md', '.py', '.sha256',
                 '.png', '.svg', '.html', '.pdf', '.ipynb'}
EXCLUDED_DIRS = {'catboost_info', '__pycache__', 'logs', 'log', 'notebook_runtime',
                 'matplotlib_cache', 'cache', 'data_cache', 'feature_cache', 'feature_caches'}
MANIFEST_NAME = 'snapshot_manifest.json'
RECEIPT_NAME = 'snapshot_manifest.sha256'
NOTICE_NAME = 'SNAPSHOT_README.md'
NOTICE = '''# 第三輪實驗快照

這是 round3 實驗與驗證證據的快照，不是 v3 發佈，也不變更目前推薦版本。
原 v1、v2 及 runs/versions.json 只讀取核對，沒有被本程式修改。

此快照不是獨立、免依賴的執行環境。原始競賽資料、大型 data_cache，以及既有
v1/v2 模型依賴不重複複製；其原始路徑、SHA-256、大小及可用備份位置記錄在
snapshot_manifest.json 的 external_dependencies。原始資料與可重建快取仍需另行
保留；只有經驗證存在的舊模型備份才會列入 existing_backups，不能假設原始資料
也存在於舊快照。來源程式中的 ROOT 應指向依賴完整的專案根目錄，不能直接將
此資料夾當成完全獨立的專案執行。

copied_files 對每個原檔與備份記錄 SHA-256 及 bytes。manifest 自身不包含自己的
雜湊，避免循環；snapshot_manifest.sha256 提供其獨立核對值。runs/versions.json
若被複製，是當次的上下文副本，不是這份實驗快照內的新版本索引。

封存若中斷，暫存目錄會保留供檢查；程式不做任何遞迴刪除。若已存在的快照
與目前待封存內容不一致，程式會拒絕覆寫。Kaggle Public 紀錄只是使用者回報的
實驗上下文，並非測試標籤或 Private 成績。
'''


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def inside(base, path):
    base = Path(base).resolve()
    candidate = Path(path)
    resolved = (candidate if candidate.is_absolute() else base / candidate).resolve()
    require(resolved != base and resolved.is_relative_to(base), f'Path leaves intended root: {path}')
    return resolved


def rel(path):
    return inside(ROOT, path).relative_to(ROOT.resolve()).as_posix()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def encode(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def file_record(path):
    path = inside(ROOT, path)
    require(path.is_file(), f'Required file is missing: {path}')
    return {'original_path': rel(path), 'original_absolute_path': str(path),
            'sha256': sha256(path), 'bytes': path.stat().st_size}


def verify_record(base, relative, digest, size=None):
    path = inside(base, relative)
    require(path.is_file(), f'Missing preserved file: {path}')
    if size is not None:
        require(path.stat().st_size == size, f'Size mismatch: {path}')
    require(sha256(path) == digest, f'SHA-256 mismatch: {path}')


def audit_prior_archives():
    records = {}
    for version in ['v1', 'v2']:
        folder = inside(ROOT, f'runs/{version}')
        manifest_path = folder / MANIFEST_NAME
        manifest = read(manifest_path)
        require(manifest['version'] == version, f'Wrong prior snapshot version: {version}')
        hashes = manifest['files_sha256']
        require(hashes, f'Empty prior snapshot manifest: {version}')
        total = 0
        for relative, digest in hashes.items():
            verify_record(folder, relative, digest)
            total += inside(folder, relative).stat().st_size
        records[version] = {'snapshot_manifest_path': rel(manifest_path),
                            'snapshot_manifest_sha256': sha256(manifest_path),
                            'verified_files': len(hashes), 'verified_bytes': total}
    return records


def verify_freeze_if_present():
    path = ROOT / 'outputs/round3/frozen_candidates.json'
    if not path.is_file():
        return {'present': False, 'note': 'No candidate freeze exists; this remains an experiment archive.'}
    receipt = path.with_suffix('.sha256')
    require(receipt.is_file() and receipt.read_text(encoding='ascii').strip() == sha256(path), 'Frozen-candidate receipt mismatch')
    frozen = read(path)
    fields = ['source_sha256', 'data_sha256', 'development_model_and_prediction_sha256',
              'development_evaluation_artifact_sha256']
    for field in fields:
        for relative, digest in frozen[field].items():
            verify_record(ROOT, relative, digest)
    return {'present': True, 'path': rel(path), 'sha256': sha256(path),
            'selected_candidate': frozen['selected_candidate'],
            'frozen_dependencies_verified': True}


def local_source_closure(paths):
    pending = list(paths)
    found = set()
    while pending:
        path = inside(ROOT, pending.pop())
        if path in found:
            continue
        require(path.is_file(), f'Source file is missing: {path}')
        found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
        for node in ast.walk(tree):
            modules = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module] if isinstance(node, ast.ImportFrom) and node.module else []
            for module in modules:
                candidate = ROOT / 'src' / (module.split('.')[0] + '.py')
                if candidate.is_file() and candidate.resolve() not in found:
                    pending.append(candidate)
    return found


def collect_copy_files():
    files = set()
    omitted = []
    outputs = ROOT / 'outputs/round3'
    require(outputs.is_dir(), 'Round-three output folder is missing')
    for folder, directories, names in os.walk(outputs, followlinks=False):
        parent = Path(folder)
        keep = []
        for name in directories:
            path = parent / name
            if name in EXCLUDED_DIRS or name.startswith('.') or path.is_symlink():
                omitted.append({'path': rel(path), 'reason': 'runtime/cache/log/staging directory or symbolic link'})
            else:
                keep.append(name)
        directories[:] = keep
        for name in names:
            path = parent / name
            if path.is_symlink() or name.startswith('.') or path.suffix.lower() in {'.lock', '.tmp', '.log', '.pyc'}:
                omitted.append({'path': rel(path), 'reason': 'temporary, lock, log or symbolic link'})
            elif path.suffix.lower() in CORE_SUFFIXES:
                files.add(inside(ROOT, path))
            else:
                omitted.append({'path': rel(path), 'reason': 'outside documented artifact suffixes'})
    files.update(p.resolve() for p in (ROOT / 'reports').glob('round3*') if p.is_file())
    report = ROOT / 'reports/optimization_round3_report.md'
    require(report.is_file(), 'Finish reports/optimization_round3_report.md before archiving')
    files.add(report.resolve())
    for relative in ['reports/kaggle_submission_history.json', 'runs/versions.json',
                     'README_round3.md', 'solution_round3.ipynb']:
        if (ROOT / relative).is_file():
            files.add((ROOT / relative).resolve())
    sources = set((ROOT / 'src').glob('*round3*.py')) | {Path(__file__).resolve()}
    files.update(local_source_closure(sources))
    requirements = list(ROOT.glob('requirements*.txt'))
    require(requirements, 'Requirement files are missing')
    files.update(path.resolve() for path in requirements)
    frozen = ROOT / 'outputs/round3/frozen_candidates.json'
    if frozen.is_file():
        extra_sources = [ROOT / p for p in read(frozen)['source_sha256'] if p.startswith('src/') and p.endswith('.py')]
        files.update(local_source_closure(extra_sources))
    for path in files:
        relative = rel(path)
        require(not relative.startswith('runs/') or relative == 'runs/versions.json', 'Never recursively copy previous snapshots')
        require(not relative.startswith('data_cache/'), 'Large data caches are external dependencies')
    return sorted(files), sorted(omitted, key=lambda r: r['path'])


def collect_external_dependencies(copy_files):
    copied = {rel(path) for path in copy_files}
    references = {}
    def add(path, reason, expected=None):
        path = inside(ROOT, path)
        require(path.is_file(), f'External dependency is missing: {path}')
        key = rel(path)
        if key in copied:
            return
        value = references.setdefault(key, {'reasons': set(), 'recorded_sha256': set()})
        value['reasons'].add(reason)
        if expected is not None:
            value['recorded_sha256'].add(expected)
    input_manifest = read(ROOT / 'outputs/round3/input_integrity.json')['input_sha256']
    require(set(input_manifest) == set(RAW_NAMES.values()), 'Input integrity record must contain four supplied files')
    for filename, digest in input_manifest.items():
        path = DATA_DIR / filename
        verify_record(DATA_DIR, filename, digest)
        add(path, 'original supplied competition input; not copied into any experiment snapshot', digest)
    for path in (ROOT / 'data_cache').iterdir():
        if path.is_file() and path.suffix in {'.pkl', '.json', '.npy', '.parquet'}:
            add(path, 'data/feature cache; retain or rebuild from original inputs')
    mapping_fields = {'source_sha256','data_sha256','input_artifact_sha256','output_artifact_sha256',
                      'development_model_and_prediction_sha256','development_evaluation_artifact_sha256',
                      'dependency_script_sha256','inference_source_sha256'}
    path_fields = {'model_path':'model_sha256','category_maps_path':'category_maps_sha256',
                   'known_projects_path':'known_projects_sha256','metadata_path':None,
                   'pickle_file':'pickle_sha256'}
    def walk(value, context):
        if isinstance(value, list):
            for child in value:
                walk(child, context)
        elif isinstance(value, dict):
            for field, hash_field in path_fields.items():
                if isinstance(value.get(field), str):
                    add(value[field], context, value.get(hash_field) if hash_field else None)
            for key, child in value.items():
                if key in mapping_fields and isinstance(child, dict):
                    for name, digest in child.items():
                        if not isinstance(digest, str) or len(digest) != 64:
                            continue
                        if name in RAW_NAMES:
                            add(DATA_DIR / RAW_NAMES[name], context, digest)
                        elif (ROOT / name).is_file():
                            add(name, context, digest)
                walk(child, context)
    add('outputs/submission_v2.csv', 'retained recommended v2 submission')
    # Previous release contexts are dependencies only; their models are not copied again.
    for relative in ['runs/v1/models/final_config.json','models/round2/release_config.json']:
        path = ROOT / relative
        add(path, 'previous release configuration/context')
        walk(read(path), relative)
    for path in copy_files:
        if path.suffix == '.json':
            walk(read(path), rel(path))
    for relative in ['runs/v1/snapshot_manifest.json','runs/v2/snapshot_manifest.json']:
        add(relative, 'existing prior snapshot proof')
    records = []
    for relative, reference in sorted(references.items()):
        record = file_record(relative)
        expected = sorted(reference['recorded_sha256'])
        record.update({'reasons': sorted(reference['reasons']), 'recorded_sha256': expected,
                       'recorded_hashes_match_current': all(h == record['sha256'] for h in expected),
                       'existing_backups': []})
        # Only report verified existing backups; raw data/cache backups are never implied.
        for version in ['v1','v2']:
            candidate = ROOT / 'runs' / version / relative
            if candidate.is_file() and sha256(candidate) == record['sha256']:
                record['existing_backups'].append(rel(candidate))
        if relative.startswith(('runs/v1/', 'runs/v2/')):
            record['existing_backups'].append(relative)
        records.append(record)
    return records


def make_plan():
    prior = audit_prior_archives()
    versions = file_record('runs/versions.json')
    freeze = verify_freeze_if_present()
    files, omissions = collect_copy_files()
    originals = {rel(path): file_record(path) for path in files}
    dependencies = collect_external_dependencies(files)
    return {'originals': originals, 'external_dependencies': dependencies,
            'prior_archives_verified': prior, 'versions_context': versions,
            'freeze_provenance': freeze, 'omitted_paths': omissions}


def check_current_plan(plan):
    for relative, record in plan['originals'].items():
        verify_record(ROOT, relative, record['sha256'], record['bytes'])
    for record in plan['external_dependencies']:
        verify_record(ROOT, record['original_path'], record['sha256'], record['bytes'])
    require(audit_prior_archives() == plan['prior_archives_verified'], 'A previous snapshot changed during archival')
    record = plan['versions_context']
    verify_record(ROOT, record['original_path'], record['sha256'], record['bytes'])


def verify_snapshot(folder, plan):
    manifest_path = folder / MANIFEST_NAME
    receipt = folder / RECEIPT_NAME
    require(manifest_path.is_file() and receipt.is_file(), 'Existing snapshot is incomplete; refusing overwrite')
    require(receipt.read_text(encoding='ascii').strip() == sha256(manifest_path), 'Snapshot manifest SHA receipt differs')
    manifest = read(manifest_path)
    require(manifest['snapshot_kind'] == 'round3_experiments' and manifest['is_release'] is False, 'Unexpected snapshot type')
    require(manifest['original_plan'] == plan, 'Current originals/dependencies differ from frozen experiment snapshot; refusing overwrite')
    expected_names = set(plan['originals']) | {MANIFEST_NAME, RECEIPT_NAME, NOTICE_NAME}
    actual_names = {path.relative_to(folder).as_posix() for path in folder.rglob('*') if path.is_file()}
    require(actual_names == expected_names, 'Snapshot has extra, missing, or different files; refusing overwrite')
    require(set(manifest['copied_files']) == set(plan['originals']), 'Snapshot artifact catalog differs')
    for relative, record in manifest['copied_files'].items():
        original = plan['originals'][relative]
        require(record['original_sha256'] == original['sha256'] == record['backup_sha256'], f'Original/backup SHA differs: {relative}')
        require(record['original_bytes'] == original['bytes'] == record['backup_bytes'], f'Original/backup size differs: {relative}')
        require(record['backup_path'] == f'runs/round3_experiments/{relative}', f'Unexpected backup path: {relative}')
        verify_record(folder, relative, record['backup_sha256'], record['backup_bytes'])
    require((folder / NOTICE_NAME).read_bytes() == NOTICE.encode('utf-8'), 'Snapshot dependency notice changed')
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-only', action='store_true', help='Verify an existing snapshot and original files without writing')
    args = parser.parse_args()
    plan = make_plan()
    destination = inside(ROOT, DESTINATION)
    runs = inside(ROOT, 'runs')
    require(destination.parent == runs and destination.name == 'round3_experiments', 'Unsafe snapshot destination')
    if destination.exists():
        manifest = verify_snapshot(destination, plan)
        check_current_plan(plan)
        print(json.dumps({'status':'verified_existing','snapshot':str(destination),'files':len(manifest['copied_files']),
                          'prior_archives_unchanged':True,'versions_unchanged':True},ensure_ascii=True),flush=True)
        return
    require(not args.verify_only, 'No round3_experiments snapshot exists yet')
    runs.mkdir(parents=True, exist_ok=True)
    required_bytes = sum(record['bytes'] for record in plan['originals'].values())
    require(shutil.disk_usage(runs).free > required_bytes + 16 * 1024 * 1024, 'Insufficient free space for a verified snapshot')
    stage = inside(runs, f'.round3-experiments-staging-{uuid.uuid4().hex}')
    require(stage.parent == runs and stage.name.startswith('.round3-experiments-staging-'), 'Unsafe staging directory')
    stage.mkdir()
    try:
        copied = {}
        for relative, original in sorted(plan['originals'].items()):
            source = inside(ROOT, relative)
            backup = inside(stage, relative)
            backup.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, backup)
            backup_sha, backup_size = sha256(backup), backup.stat().st_size
            require(backup_sha == original['sha256'] and backup_size == original['bytes'], f'Copy verification failed: {relative}')
            copied[relative] = {'original_path':relative,'original_absolute_path':str(source),
                                'original_sha256':original['sha256'],'original_bytes':original['bytes'],
                                'backup_path':f'runs/round3_experiments/{relative}',
                                'backup_sha256':backup_sha,'backup_bytes':backup_size}
        manifest = {'format_version':1,'snapshot_kind':'round3_experiments','is_release':False,
                    'release_version':None,'created_at_utc':datetime.now(timezone.utc).isoformat(),
                    'self_contained':False,'dependency_notice':NOTICE_NAME,
                    'copied_files':copied,'external_dependencies':plan['external_dependencies'],
                    'original_plan':plan,'prior_archives_modified':False,'versions_modified':False,
                    'raw_data_and_large_caches_duplicated':False,'prior_release_models_duplicated':False,
                    'manifest_self_excluded':True,'manifest_sha256_receipt':RECEIPT_NAME,
                    'failure_policy':'Retain staging for inspection; never recursively delete or overwrite a different snapshot.'}
        (stage / NOTICE_NAME).write_bytes(NOTICE.encode('utf-8'))
        (stage / MANIFEST_NAME).write_bytes(encode(manifest))
        (stage / RECEIPT_NAME).write_text(sha256(stage / MANIFEST_NAME) + '\n', encoding='ascii')
        verify_snapshot(stage, plan)
        check_current_plan(plan)
        require(not destination.exists(), 'Snapshot appeared during copy; refusing overwrite')
        require(stage.resolve().parent == runs and destination.resolve().parent == runs and stage.resolve().is_relative_to(ROOT.resolve()), 'Unsafe snapshot move')
        stage.rename(destination)
    except BaseException:
        print(f'Archive did not finalize. Any staging files are retained at: {stage}', flush=True)
        raise
    verify_snapshot(destination, plan)
    print(json.dumps({'status':'created','snapshot':str(destination),'files':len(manifest['copied_files']),
                      'copied_bytes':required_bytes,'external_dependencies':len(plan['external_dependencies']),
                      'snapshot_manifest_sha256':sha256(destination / MANIFEST_NAME),
                      'is_release':False,'self_contained':False,
                      'prior_archives_unchanged':True,'versions_unchanged':True},ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()