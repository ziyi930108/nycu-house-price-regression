"""Bounded R114--R117 four-origin evidence package; standard library only.

Never fit, score, load arrays/models, run git/network, or publish. --check is
read-only; --apply and --refresh-manifest are parent-only explicit actions after
independent source review. Missing preflight/postfit/completion is a hard failure.
Only individually named aggregate evidence is eligible; opaque hashes of private
artifacts can be checked without copying or deserializing their bytes.

This revision accepts failed no-release or eligible-but-deferred dispositions.
A completed future full release needs a separately reviewed extension; it is not
silently admitted without exact final-release/CSV authoring review.

Original f596c42 base:3229manifest/3230tracked, all3226non-navigation members,
including historical379research/13production and former3014protected, preserved.
Navigation may add only dated prefix/suffix preserving all original body bytes.
Partial-copy failure leaves evidence and refuses a silent rerun; not atomic.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
import stat
import struct
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'deliverables/v22_github'
ROUNDS = (114,116,117)
VERSIONS = {114:30,116:31,117:32}
STAGES = ('development','confirmation','auxiliary','offset_2023_02_11')
GROUPS = tuple(f'outputs/research_round{n}_20261008' for n in (114,115,116,117)) + ('outputs/research_continue_20261008','outputs/research_round118_feasibility_20261008')
SNAPSHOT = 'evidence/outputs/research_package_four_origins_20261008'
COPY_MANIFEST = SNAPSHOT + '/copy_manifest.json'
SUMMARY = 'docs/RESEARCH_FOUR_ORIGINS_20261008.md'
NAVIGATION = ('README.md','research/README.md','HANDOFF.md')
BASE_HEAD = 'f596c4260d10144bdbc963d1c2a57519375a7ba0'
BASE_MANIFEST_SHA256 = '9112baad885108e19fba1fc3ac6e60d8045a2786533cbc1c8482a345b4253000'
BASE_MANIFEST_COUNT = 3229
BASE_TRACKED_COUNT = 3230
ALLOWED_SUFFIXES = {'.py','.md','.json','.sha256','.log'}
FORBIDDEN_PARTS = {'node_modules','__pycache__','.git','.venv','venv','rawinput','raw_input','rawinputs','frames','private','runs'}
FORBIDDEN_NAMES = {'feature_state.json','state.json','state_summary.json','price_prior.json','price_prior_state.json','prior_state.json','encoder_state.json','prior_evidence.json','preparation.json','csv_authoring_receipt.json','preview.json','inspection.json'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})')
IMPLEMENTATION_PARENT = {'path':'src/package_research_continue_20261008_v2.py','sha256':'a9071908de685677bcbd1bc726d983c6fe5052f953afd1ea23bbbce587858b54'}
PRODUCTION = (
 'src/house_features.py','src/nn_predict.py','src/nn_prepare.py','src/predict_v22.py',
 'src/round47_nn_encode.py','src/round54_numeric_encode.py','src/submission_io.py','src/tree_runtime.py',
 'src/closure_modules/model_common.py','src/closure_modules/prepare_data.py',
 'src/closure_modules/round2_neighborhood.py','src/closure_modules/round6_geo.py','src/closure_modules/round8_spatial_reference.py')
SOURCE_FILES = (
 'src/round114_research.py','src/round114_additive_prior_train.py','src/summarize_round114.py',
 'src/round115_prior_support_x_audit.py','src/summarize_round115.py',
 'src/round116_research.py','src/round116_depthwise_unit.py','src/summarize_round116.py',
 'src/round117_research.py','src/round117_transaction_prior.py','src/round117_transaction_prior_train.py',
 'src/summarize_round117.py','src/package_research_four_origins_20261008.py')
REPORT_FILES = (
 'reports/round114_plan.md','reports/round114_report.md','reports/round114_method_review.md',
 'reports/round115_plan.md','reports/round115_report.md',
 'reports/round116_plan.md','reports/round116_report.md','reports/round116_method_review.md','reports/round116_driver_review.md',
 'reports/round117_plan.md','reports/round117_report.md','reports/round117_method_review.md','reports/round117_driver_review.md')
METHODS = {114:'先驗金額＋signed NN修正；weighted總價MAE，推論一次非負投影',
 116:'單價CatBoost只加入Depthwise；Plain/date×area保持',
 117:'fit先驗改固定交易五折；原Softplus倍率NN／其他八array／query先驗保持'}
FEASIBILITY_GROUP='outputs/research_round118_feasibility_20261008'
FEASIBILITY_ARTIFACTS=('audit_feasibility.py','author_synthetic_checks.json','design_limits.json',
 'earliest_boundary_feasibility.json','earliest_boundary_request.json','earliest_local_boundary.py',
 'feasibility.json','feasibility.md','final_design_limits.json','finalize_evidence.py',
 'packaging_failure_attempt1.json','partial_band_feasibility.json','planned_readset.json',
 'readset.json','seal_evidence.py','supplemental_date_candidates.py','supplemental_request.json')


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha(path):
    with path.open('rb') as handle:
        digest = hashlib.sha256()
        for block in iter(lambda: handle.read(1 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def relative(name):
    need(isinstance(name, str) and '\\' not in name and ':' not in name, 'Use relative POSIX paths')
    need(SECRET.search(name) is None, 'Possible credential in artifact path')
    path = PurePosixPath(name)
    need(bool(path.parts) and name == path.as_posix() and not path.is_absolute() and '..' not in path.parts,
         'Unsafe/noncanonical relative path: ' + name)
    need(not any(part.lower() in FORBIDDEN_PARTS for part in path.parts), 'Excluded directory: ' + name)
    return path


def reject_reparse(path, root):
    """Do not follow symlink/junction ancestors, including planned destinations."""
    current = path
    while current.is_relative_to(root):
        if current.exists() or current.is_symlink():
            info = current.lstat()
            need(not current.is_symlink()
                 and not (getattr(info, 'st_file_attributes', 0)
                          & getattr(stat, 'FILE_ATTRIBUTE_REPARSE_POINT', 0)),
                 'Symlink/reparse point requires separate review: ' + str(current))
        if current == root:
            break
        current = current.parent


def safe(root, name):
    relative(name)
    root = root.absolute()
    candidate = root / name
    reject_reparse(candidate, root)
    path = candidate.resolve()
    need(path.is_relative_to(root.resolve()), 'Path escapes intended directory: ' + name)
    return path


def reject_nonfinite(value):
    raise ValueError('Nonfinite JSON scalar: ' + value)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        need(key not in result, 'Duplicate JSON key: ' + key)
        result[key] = value
    return result


def json_value(payload):
    return json.loads(payload.decode('utf-8-sig'), object_pairs_hook=unique_object,
                      parse_constant=reject_nonfinite)


def check_text(payload, label):
    """Apply the same credential guard to copied and newly generated text."""
    need(len(payload) <= 32 * 1024 * 1024, 'Unexpectedly large evidence file: ' + label)
    text = payload.decode('utf-8-sig')
    need(SECRET.search(text) is None, 'Possible credential in selected/generated evidence: ' + label)
    return text


def check_aggregate(value, key='$'):
    """Only aggregate evidence, never embedded rows, prices or learned state."""
    blocked_payloads = {'inspection', 'inspection_ndjson', 'preview', 'preview_rows', 'raw_preview',
        'serialized_rows', 'csv_text', 'csv_content', 'row_values', 'state_dict', 'cells',
        'embedding_weights', 'tensor_values', 'feature_values', 'prior_by_key', 'encoder_mapping'}
    need(not ((key.lower() in blocked_payloads or re.search(r'(?:^|_)(?:preview|inspection|ndjson)(?:_|$)',key.lower()))
              and isinstance(value, (dict, list, str))),
         'Embedded row/learned-state payload excluded: ' + key)
    if isinstance(value, dict):
        lower = {k.lower() for k in value}
        need(not (lower & {'id', 'sample_id', 'canonical_index', 'row_index', '編號'}
                  and lower & {'price', 'price_twd', 'prediction', 'predicted_price', 'target', '總價元'}),
             'Per-row identity and price record excluded')
        for child, item in value.items():
            check_aggregate(item, child)
    elif isinstance(value, list):
        need(key.lower() not in {'rows', 'csv_rows', 'raw_rows', 'frame_rows', 'data_rows',
                  'prices', 'predictions', 'targets', 'labels', 'ids', 'values', 'sample_values', '編號', '總價元'},
             'Embedded per-row vector excluded: ' + key)
        need(not (len(value) >= 256 and all(isinstance(item, (int, float)) for item in value)),
             'Long numeric vector excluded')
        for item in value:
            check_aggregate(item, key)
    elif isinstance(value, float):
        need(math.isfinite(value), 'Nonfinite aggregate scalar')


def destination(name):
    path = policy(name)
    return ('research/' if path.parts[0] == 'src' else 'evidence/') + name


def read(name):
    payload = safe(ROOT, name).read_bytes()
    policy(name, payload)
    return json_value(payload)


def reference(name):
    """Opaque file hash only; private referenced artifacts are never copied/decoded."""
    path = safe(ROOT, name)
    need(path.is_file(), 'Missing bound artifact: ' + name)
    return {'path': name, 'sha256': sha(path)}


def check_record(record, expected_path=None):
    need(isinstance(record, dict) and set(record) == {'path', 'sha256'}, 'Expected exact path/SHA record')
    if expected_path is not None:
        need(record['path'] == expected_path, 'Artifact reference points to unexpected file')
    need(record == reference(record['path']), 'Artifact reference SHA mismatch: ' + record['path'])
    return record


def git_index_paths(payload):
    """Parse only local SHA1 Git index v2/v3; never run a git command."""
    need(len(payload) >= 32 and hashlib.sha1(payload[:-20]).digest() == payload[-20:],
         'Git index checksum differs')
    signature, version, count = struct.unpack('!4sII', payload[:12])
    need(signature == b'DIRC' and version in (2, 3), 'Expected Git index v2/v3')
    offset, names = 12, []
    for _ in range(count):
        start = offset
        need(offset + 62 < len(payload)-20, 'Truncated index entry')
        flags = struct.unpack('!H', payload[offset+60:offset+62])[0]
        need(not flags & 0x3000, 'Unmerged/staged index entry is excluded')
        offset += 62
        if flags & 0x4000:
            need(version == 3, 'Extended index entry requires v3')
            offset += 2
        end = payload.find(b'\0', offset)
        need(end >= offset and end < len(payload)-20, 'Missing index path terminator')
        name = payload[offset:end].decode('utf-8')
        relative(name)
        need((flags & 0xfff) in (min(len(payload[offset:end]), 0xfff),), 'Index name length differs')
        names.append(name)
        offset = start + ((end + 1 - start + 7)//8)*8
    need(len(names) == len(set(names)), 'Duplicate tracked index path')
    return names


def check_git_base(expected_members=None):
    """Read HEAD/ref/index, no git/process/network or private traversal."""
    git = REPO / '.git'
    reject_reparse(git,REPO)
    need(git.is_dir() and not git.is_symlink(), 'Expected original standalone repository .git directory')
    head = (git/'HEAD').read_text(encoding='ascii').strip()
    if head.startswith('ref: '):
        ref = head[5:]
        need(re.fullmatch(r'refs/[A-Za-z0-9_./-]+', ref) and '..' not in PurePosixPath(ref).parts,
             'Invalid symbolic HEAD')
        loose = git/ref
        if loose.is_file():
            head = loose.read_text(encoding='ascii').strip()
        else:
            packed = (git/'packed-refs').read_text(encoding='ascii').splitlines()
            candidates = [line.split(' ')[0] for line in packed if line.endswith(' '+ref) and not line.startswith(('#','^'))]
            need(len(candidates) == 1, 'Cannot uniquely resolve packed HEAD')
            head = candidates[0]
    need(head == BASE_HEAD, 'Repository HEAD changed from approved base; review before packaging')
    names = git_index_paths((git/'index').read_bytes())
    need(len(names) == BASE_TRACKED_COUNT, 'Original tracked index count changed')
    if expected_members is not None:
        need(set(names) == set(expected_members) | {'SOURCE_MANIFEST.json'},
             'Tracked membership differs from original explicit manifest')
    return {'HEAD': head, 'index_tracked_members': len(names), 'git_commands_run': 0}


def check_sidecar(name,payload):
    text=payload.decode('ascii').strip()
    match=re.fullmatch(r'([0-9a-f]{64})(?:[ \t]+([^\r\n]+))?',text)
    need(match is not None,'Expected one SHA record; multi-file sidecars require separate review')
    digest,filename=match.groups();p=PurePosixPath(name)
    if filename is not None:
        need(PurePosixPath(filename).name==filename and '\\' not in filename,
             'SHA sidecar may only name a same-directory basename')
        counterpart=str(p.parent/filename);policy(counterpart)
        need(safe(ROOT,counterpart).is_file() and sha(safe(ROOT,counterpart))==digest,
             'Named SHA sidecar differs')
        need(counterpart in {name[:-7],str(p.with_suffix('.json')),str(p.with_suffix('.md')),
                             str(p.with_suffix('.py')),str(p.with_suffix('.log'))},
             'Sidecar does not bind its corresponding basename')
    else:
        candidates={name[:-7],str(p.with_suffix('.json')),str(p.with_suffix('.md')),
                    str(p.with_suffix('.py')),str(p.with_suffix('.log'))}
        found=[x for x in candidates if safe(ROOT,x).is_file() and sha(safe(ROOT,x))==digest]
        need(len(found)==1,'Bare SHA must bind one unambiguous counterpart')


def package_files(additions=(), template=None):
    """Hash only old manifest members plus explicit additions, never private/runs.

    The local checkout legitimately contains ignored private models and replay
    outputs. Filesystem-wide discovery would publish those in SOURCE_MANIFEST.
    Its old explicit list, rather than os.walk/rglob or git, defines the scope.
    """
    if template is None:
        template = json_value((REPO / 'SOURCE_MANIFEST.json').read_bytes())
    need(isinstance(template['files'], list) and len({item['path'] for item in template['files']}) == len(template['files']),
         'Published manifest contains duplicate/non-list entries')
    names = {item['path'] for item in template['files']} | set(additions) | {'SOURCE_MANIFEST.json'}
    records = {}
    private_suffixes = {'.npz', '.npy', '.pt', '.cbm', '.pkl', '.pickle', '.csv', '.xlsx', '.xls', '.parquet', '.pyc'}
    for name in sorted(names):
        relative_path = relative(name)
        need(relative_path.parts[0] not in {'private', 'runs'} and relative_path.suffix.lower() not in private_suffixes,
             'Private/runtime file cannot be a published manifest member: ' + name)
        unresolved = REPO / name
        for path in (unresolved,) + tuple(unresolved.parents):
            if path == REPO.parent:
                break
            need(not path.is_symlink() and not (hasattr(path, 'is_junction') and path.is_junction()),
                 'Package symlink/junction needs separate review: ' + str(path))
        path = safe(REPO, name)
        need(path.is_file(), 'Required package member missing: ' + name)
        records[name] = {'path': name, 'bytes': path.stat().st_size, 'sha256': sha(path)}
    return records


def protect_existing(records):
    old_bytes = (REPO / 'SOURCE_MANIFEST.json').read_bytes()
    need(sha_bytes(old_bytes) == BASE_MANIFEST_SHA256, 'Original 3229-member manifest identity changed; review before packaging')
    old = json_value(old_bytes)
    check_git_base(item['path'] for item in old['files'])
    need(len(old['files']) == BASE_MANIFEST_COUNT, 'Original package membership count changed')
    expected = {item['path']: item for item in old['files']}
    for name, record in expected.items():
        need(name in records and records[name] == record, 'Pre-existing manifest member differs: ' + name)
    need(all(name in expected for name in PRODUCTION + ('research/source_manifest.json',)),
         'Original package lacks a required production/historical member')
    history = json_value((REPO / 'research/source_manifest.json').read_bytes())
    need(isinstance(history, list) and len(history) == 379, 'Historical 379-source manifest changed')
    for item in history:
        need(records[item['copy']]['sha256'] == item['sha256'], 'Historical source copy changed: ' + item['copy'])
    return old, {name: item['sha256'] for name, item in records.items()
                 if name != 'SOURCE_MANIFEST.json' and name not in NAVIGATION}


def write_new(path, payload):
    check_text(payload, str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('xb') as handle:
        handle.write(payload)
    need(path.read_bytes() == payload, 'Written bytes differ: ' + str(path))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n').encode('utf-8')


def refresh_top(template, additions=()):
    current = package_files(additions, template)
    replacement = dict(template)
    replacement['created_at_utc'] = datetime.now(timezone.utc).isoformat()
    replacement['files'] = [current[name] for name in sorted(current) if name != 'SOURCE_MANIFEST.json']
    replacement['private_artifacts_included'] = False
    replacement['raw_course_data_included'] = False
    replacement['research_four_origins_20261008'] = {'copy_manifest': COPY_MANIFEST, 'publication_not_performed': True}
    target = REPO / 'SOURCE_MANIFEST.json'
    temporary = target.with_name('SOURCE_MANIFEST.four_origins_20261008.tmp')
    write_new(temporary, encoded(replacement))
    os.replace(temporary, target)
    actual = package_files()
    need({item['path']: item for item in replacement['files']}
         == {name: item for name, item in actual.items() if name != 'SOURCE_MANIFEST.json'},
         'Top-level manifest does not cover exact explicitly selected package members')
    return {'files': len(replacement['files']), 'sha256': sha(target)}


def verify_protected(protected):
    for name, expected in protected.items():
        need(sha(safe(REPO, name)) == expected, 'Pre-existing protected package file changed: ' + name)


def check_navigation_note(original, current, name):
    need(current.startswith(original) or current.endswith(original), 'Navigation must preserve original body bytes: ' + name)
    addition = current[len(original):] if current.startswith(original) else current[:-len(original)]
    check_text(current, name)
    need(len(addition) <= 65536 and (not addition or '2026-10-08' in addition.decode('utf-8-sig')),
         'Navigation addition must be a bounded 2026-10-08 dated note: ' + name)


def finish_manifest():
    check_git_base()
    template = json_value((REPO / 'SOURCE_MANIFEST.json').read_bytes())
    current_records = package_files(template=template)
    listed = {item['path']: item for item in template['files']}
    need(COPY_MANIFEST in listed and current_records[COPY_MANIFEST] == listed[COPY_MANIFEST],
         'Copy manifest changed after packaging')
    receipt = json_value((REPO / COPY_MANIFEST).read_bytes())
    original_manifest = (REPO / SNAPSHOT / 'before/SOURCE_MANIFEST.json').read_bytes()
    need(sha_bytes(original_manifest) == BASE_MANIFEST_SHA256, 'Archived original manifest identity changed')
    old = json_value(original_manifest)
    expected_names = ({item['path'] for item in old['files']}
                      | {item['destination'] for item in receipt['files']}
                      | {item['path'] for item in receipt['generated']} | {COPY_MANIFEST})
    need(set(listed) == expected_names, 'Unexpected/missing top-manifest member after packaging')
    for name, expected in listed.items():
        if name not in NAVIGATION:
            need(current_records[name] == expected, 'Manifest-bound package member changed before finish: ' + name)
    verify_protected(receipt['protected_before'])
    for item in receipt['files']:
        need(sha(safe(REPO, item['destination'])) == item['sha256'], 'Packaged source/evidence changed')
    for item in receipt['generated']:
        need(sha(safe(REPO, item['path'])) == item['sha256'], 'Generated document/projection/archive changed')
    for name in NAVIGATION:
        original = (REPO / SNAPSHOT / 'before' / name).read_bytes()
        current = (REPO / name).read_bytes()
        check_navigation_note(original, current, name)
    finish_path = REPO / SNAPSHOT / 'finish_manifest_receipt.json'
    need(not finish_path.exists(), 'Preserve an existing finish receipt')
    finish = {'status': 'dated_navigation_finish_local_only', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'navigation': {name: sha(REPO / name) for name in NAVIGATION},
              'copy_manifest_sha256': sha(REPO / COPY_MANIFEST), 'protected_files_unchanged': True,
              'git_or_network': False}
    write_new(finish_path, encoded(finish))
    return {**finish, 'SOURCE_MANIFEST': refresh_top(template, [finish_path.relative_to(REPO).as_posix()])}


def policy(name, payload=None):
    path = relative(name)
    need(path.suffix.lower() in ALLOWED_SUFFIXES and path.name.lower() not in FORBIDDEN_NAMES,
         'Excluded artifact type/private state: ' + name)
    need(path.parts[0] in {'src','reports','outputs'}, 'Only bounded named evidence')
    if path.parts[0] == 'src':
        need(len(path.parts) == 2 and path.suffix == '.py', 'Standalone Python source only')
    elif path.parts[0] == 'reports':
        need(len(path.parts) == 2 and re.fullmatch(r'round11[4-8][A-Za-z0-9_.-]*', path.name),
             'Only R114--R118 reports are in scope')
    else:
        need(len(path.parts) >= 3 and '/'.join(path.parts[:2]) in GROUPS + tuple(
            f'outputs/round{n}' for n in (114,115,116,117,118)), 'Output outside bounded roots')
        need('prior' not in path.parts[2:] and 'prepared' not in path.parts[2:],
             'Private prior/prepared folder excluded, including JSON')
        need(not re.search(r'(?:_rows|_preview|_inspection)\.(?:json|log)$',path.name,re.I),
             'Per-row/preview evidence excluded')
    if payload is not None:
        text = check_text(payload,name)
        if path.suffix == '.json':
            check_aggregate(json_value(payload))
        elif path.suffix == '.log':
            for line in text.splitlines():
                need(len(line) <= 65536 and not re.fullmatch(r'\s*\d+\s*,\s*-?\d+(?:\.\d+)?\s*',line),
                     'Per-row or oversized log line excluded: ' + name)
                if line.lstrip().startswith(('{','[')):
                    try:
                        value = json_value(line.encode('utf-8'))
                    except (json.JSONDecodeError,UnicodeError):
                        continue
                    check_aggregate(value)
    return path


def opaque_path(name):
    """Canonical private references may include runs/, but never become copies."""
    need(isinstance(name,str) and '\\' not in name and ':' not in name
         and SECRET.search(name) is None,'Unsafe opaque path')
    p=PurePosixPath(name)
    need(name==p.as_posix() and p.parts and not p.is_absolute() and '..' not in p.parts,
         'Opaque reference escapes workspace')
    path=ROOT/name
    reject_reparse(path,ROOT)
    need(path.resolve().is_relative_to(ROOT.resolve()) and path.is_file(),
         'Missing/escaping opaque artifact: '+name)
    return path


def check_hashes(mapping, label):
    need(isinstance(mapping,dict) and mapping, 'Missing exact hash map: '+label)
    for name,expected in mapping.items():
        need(isinstance(expected,str) and re.fullmatch(r'[a-f0-9]{64}',expected), 'Invalid SHA in '+label)
        need(sha(opaque_path(name)) == expected,'Bound bytes changed: '+name)


def check_opaque_record(record):
    need(isinstance(record,dict) and set(record)=={'path','sha256'},'Exact opaque path/SHA required')
    need(sha(opaque_path(record['path']))==record['sha256'],'Opaque artifact SHA mismatch')


def collect_record(record, names):
    check_record(record)
    policy(record['path'])
    names.add(record['path'])


def check_identity(record):
    need(isinstance(record,dict) and {'path','sha256'}.issubset(record),'Artifact identity missing')
    check_opaque_record({k:record[k] for k in ('path','sha256')})


def require_common_review(review, number):
    need(review['status']=='passed' and review['round']==number
         and type(review['new_model_fits']) is int and review['new_model_fits']==0 and review['GPU_work'] is False
         and review['Private_score'] is None and review['Public_score'] is None
         and review['new_CSV'] is False and review['blocking_findings_remaining']==[],
         'Independent postfit is incomplete or contains blocking findings')


def manual_gates(gates, extra_both_improve):
    august=gates['august']; extra=gates['extra_origin']
    return {'august':{
        'components':august['exploratory_submission']['components'],
        'exploratory_submission_eligible':august['exploratory_submission']['passed'],
        'strong_components':august['promotion']['components'],
        'strong_research_gate_passed':august['promotion']['adoption_research_gate_passed'],
        'confirmation_fractional_improvement':august['promotion']['confirmation_fractional_improvement']},
        'extra_origin':{**extra,'both_point_metrics_improve':extra_both_improve},
        'combined_exploratory_submission':gates['combined_exploratory_submission'],
        'combined_promotion':gates['combined_promotion']}


def check_combined(gates):
    a=gates['august']; e=gates['extra_origin']
    pairs=(('combined_exploratory_submission',a['exploratory_submission']['passed']),
           ('combined_promotion',a['promotion']['adoption_research_gate_passed']))
    for key,august_passed in pairs:
        item=gates[key]
        need(type(august_passed) is bool and type(e['passed']) is bool
             and item=={'components':{'original_three_windows':august_passed,
                 'fixed_extra_origin':e['passed']},'passed':august_passed and e['passed']},
             'Combined gate must be exact August AND extra origin')
    for node in (a['exploratory_submission'],e):
        need(node['components'] and all(type(v) is bool for v in node['components'].values())
             and node['passed'] is all(node['components'].values()), 'Exploration component verdict differs')
    p=a['promotion']
    need(p['components'] and all(type(v) is bool for v in p['components'].values())
         and p['adoption_research_gate_passed'] is all(p['components'].values()),'Strong component verdict differs')


def parse_dispositions(no_releases,deferred_releases):
    selected={}
    for key,items in (('no_release_reason',no_releases),('deferred_release_reason',deferred_releases)):
        for item in items:
            left,sep,right=item.partition('=')
            need(sep and left.isdigit() and int(left) in ROUNDS and len(right.strip())>=8,
                 'Supply ROUND=concrete reason for 114/116/117')
            n=int(left);need(n not in selected,'Duplicate round disposition')
            check_text(right.encode('utf-8'),'disposition')
            selected[n]={key:right.strip()}
    need(set(selected)==set(ROUNDS),'Explicit disposition required for each114/116/117')
    return selected


def parse_postfit_executions(items):
    result={}
    for item in items:
        left,sep,right=item.partition('=')
        need(sep and left.isdigit() and int(left) in (114,115,116,117),
             'Postfit receipt override must name ROUND=EXACT_PATH')
        n=int(left);need(n not in result,'Duplicate postfit receipt override')
        p=policy(right)
        need(str(p.parent)==f'outputs/research_round{n}_20261008'
             and re.fullmatch(r'independent_postfit_execution(?:_v\d+)?\.json',p.name),
             'Postfit receipt must be own-round named execution JSON')
        result[n]=right
    return result


def check_disposition(number,disposition,result):
    check_combined(result['gates'])
    eligible=result['gates']['combined_exploratory_submission']['passed']
    key='deferred_release_reason' if eligible else 'no_release_reason'
    need(set(disposition)=={key} and len(disposition[key])>=8,
         'Gate pass requires eligible-deferred; failed combined requires no-release')
    check_text(disposition[key].encode('utf-8'),'disposition')
    need(not safe(ROOT,f'outputs/round{number}/final/release.json').exists(),
         'Future completed full release needs separately reviewed packaging schema')
    need(not any((ROOT/'outputs').glob(f'submission_v{VERSIONS[number]}*.csv')),
         'No-release/defer conflicts with existing CSV')
    return 'eligible_full_release_deferred' if eligible else 'combined_gate_failed_no_release'


def check_execution(name,entry,flag,names):
    value=read(name);command=value['command']
    need(value['exit_code']==0 and isinstance(command,list)
         and command[-1]==flag and any(str(x).replace('\\','/').endswith('/'+entry) for x in command)
         and isinstance(value['elapsed_seconds'],(int,float)) and value['elapsed_seconds']>=0,
         'Formal command did not successfully execute expected entry/flag: '+name)
    names.add(name)
    return value


def check_model_metadata(number,stage,item,meta,request):
    need(meta['status']=='completed' and meta['saved_model_reload_exact'] is True
         and meta['query_labels_used'] is False and meta['fit_rows']==item['fit_rows']
         and meta['query_rows']==item['query_rows'] and meta['reload_rows']==item['query_rows'],
         'Model is incomplete or full-query reload unverified')
    if number==116:
        need(meta['raw_and_TWD_reload_exact'] is True and meta['actual_grow_policy']=='Depthwise'
             and meta['actual_boosting_type']=='Plain' and meta['actual_task_type']=='GPU'
             and meta['actual_ctr_target_border_count']==1 and meta['actual_max_ctr_complexity']==2
             and meta['actual_trees']==2999 and meta['actual_model_depth']==8
             and meta['actual_seed']==42 and meta['actual_min_data_in_leaf']==1
             and meta['tree_leaf_counts_count']==2999
             and sum(meta['tree_leaf_counts_distribution'].values())==2999
             and all(k.isdigit() and 1<=int(k)<=256 and type(v) is int and v>0
                     for k,v in meta['tree_leaf_counts_distribution'].items()),'Depthwise native contract differs')
    else:
        need(meta['configuration']==request['recipe']['configuration']
             and [x['epoch'] for x in meta['epochs']]==list(range(1,31)), 'NN frozen configuration/epochs differ')
        if number==114:
            need(meta['saved_raw_total_reload_exact'] is True and meta['training_total_clipped'] is False,
                 'Signed additive NN must preserve raw total/reload')
        else:
            need(meta['saved_ratio_reload_exact'] is True
                 and meta['actual_model_class']=='round47_nn_train.EmbeddingMLP'
                 and meta['changed_configuration_fields']==['train_price_factor_construction']
                 and meta['ML_configuration_changes']==[] and meta['loss_configuration_changes']==[],
                 'Transaction prior NN must retain original architecture/loss')
            need(item['prepared_array_identities']==meta['input_arrays'], 'Prepared/model nine-array identity differs')


def sealed_model_set(number,prefix,records):
    required={114:{'model.pt','run.json','training.json','predictions_raw_twd.npy','predictions_twd.npy'},
        116:{'feature_state.json','metadata.json','model.cbm','native.npy','raw.npy','started.json'},
        117:{'model.pt','run.json','training.json','ratio_predictions.npy','predictions_twd.npy'}}[number]
    need(isinstance(records,list) and len(records)==len(required)
         and all(isinstance(x,dict) and set(x)=={'path','sha256'} for x in records)
         and {PurePosixPath(x['path']).name for x in records}==required
         and all(str(PurePosixPath(x['path']).parent)==prefix+'/model' for x in records),
         'Complete exact own-origin model artifact set required')


def postfit_execution(number,post,names,overrides):
    group=f'outputs/research_round{number}_20261008'
    default=group+'/independent_postfit_execution'+('_v2' if number==116 else '')+'.json'
    name=overrides.get(number,default);p=policy(name)
    need(str(p.parent)==group and re.fullmatch(r'independent_postfit_execution(?:_v\d+)?\.json',p.name),
         'Postfit command receipt must be an explicit own-round execution record')
    value=read(name);names.add(name)
    need(value['exit_code']==0 and value['automatic_retry'] is False and value['GPU_work'] is False
         and value['postfit_review']==reference(group+'/postfit_review.json')
         and value.get('new_model_fits',value.get('new_fits'))==0
         and math.isfinite(value['elapsed_seconds']) and value['elapsed_seconds']>=0
         and any(str(x).replace('\\','/').endswith('/'+post['review_tool']['path']) for x in value['command']),
         'Actual independent CPU postfit command incomplete or unbound')
    collect_record(value['launcher'],names);collect_record(value['log'],names)
    if number==115:
        need(value['review_tool_before']==value['review_tool_after']==post['review_tool']
             and value['query_y_or_housing_arrays_deserialized'] is False,
             'Pure-X independent execution changed source/read forbidden data')
    else:
        need(value['failure'] is None and value['review_tool']==post['review_tool']
             and value['evaluation_execution']==post['evaluation_execution'], 'Independent command exact tool/eval differs')


def interpretation_binding(number,names):
    """False prediction-only flags never imply scored raw MAE did not read y."""
    group=f'outputs/research_round{number}_20261008'
    binding=read(group+'/postfit_interpretation_binding.json')
    need(binding['status']=='passed' and binding['interpretations_consistent'] is True
         and binding['immutable_results_and_source_preserved'] is True
         and binding['query_y_read'] is False and binding['new_fits']==0
         and binding['GPU_work'] is False and binding['fullfit_release_newCSV'] is False
         and binding['Public_score'] is None and binding['Private_score'] is None,
         'Missing explicit immutable evaluation-semantics binding')
    for field,path in [('evaluation_results',f'outputs/round{number}/results.json'),
        ('postfit_review',group+'/postfit_review.json'),
        ('root_semantics_clarification',group+'/evaluation_semantics_clarification.json'),
        ('independent_interpretation',group+'/postfit_interpretation_review.json')]:
        need(binding[field]==reference(path),'Interpretation exact binding differs')
        collect_record(binding[field],names)
    root_note=read(binding['root_semantics_clarification']['path'])
    note=read(binding['independent_interpretation']['path'])
    need(root_note['overall_mae_uses_query_labels'] is True
         and root_note['overall_mae_computed_only_in_post_seal_evaluation'] is True
         and root_note['original_results_source_and_gates_unchanged'] is True,
         'Root semantic scope is inconsistent')
    if number==114:
        need(note['status']=='passed_with_explicit_semantic_scope'
             and note['scientific_metrics_and_manual_gate_verdicts_unchanged'] is True
             and note['query_y_reread_by_this_interpretation_note'] is False
             and note['interpretation']['raw_signed_overall_mae_uses_query_y'] is True
             and note['interpretation']['query_y_read_only_after_all_four_seals_and_formal_training_success'] is True
             and note['postfit_review']==binding['postfit_review']
             and note['evaluation_results']==binding['evaluation_results'], 'R114 semantic scope differs')
    else:
        post=read(group+'/postfit_review.json')
        need(note['status']=='passed' and note['round']==116
             and note['unprojected_TWD_MAE_uses_query_y'] is True
             and note['original_source_results_gates_receipts_unmodified'] is True
             and note['scientific_metrics_and_verdicts_unchanged'] is True
             and note['scoring_authorized_only_after_all_four_models_sealed_and_training_exit0'] is True
             and note['source_sha256']==post['source_sha256']
             and note['new_model_fits']==0 and note['GPU_work'] is False
             and note['Public_score'] is None and note['Private_score'] is None
             and note['selected_submission']=='v22' and note['new_CSV'] is False,
             'R116 independent semantic scope differs')
        for field in ('request','seal','results','evaluation_execution','postfit_review','review_tool'):
            expected=reference(group+'/postfit_review.json') if field=='postfit_review' else post[field]
            need(note[field]==expected,'R116 semantic exact binding differs: '+field)
    names.add(group+'/postfit_interpretation_binding.json')


def model_round(number,disposition,names,postfit_overrides):
    folder=f'outputs/round{number}';group=f'outputs/research_round{number}_20261008'
    paths={k:folder+'/'+k+'.json' for k in ('request','seal','results')}
    paths.update({k:group+'/'+k+'.json' for k in ('preflight_review','postfit_review','training_execution','evaluation_execution')})
    names.update(paths.values())
    req,seal,result,pre,post=(read(paths[k]) for k in ('request','seal','results','preflight_review','postfit_review'))
    need(req['round']==number and req['new_fits']==4 and req['origin_order']==list(STAGES)
         and req['public_tuning'] is False and req['query_scores_computed'] is False
         and req['query_labels_used_for_fit'] is False and req['new_CSV'] is False
         and req['Public_score'] is None and req['Private_score'] is None and req['selected_submission']=='v22',
         'Frozen request scope differs')
    collect_record(req['plan'],names)
    need(req['plan']['path']==f'reports/round{number}_plan.md','Wrong fixed plan')
    sources=req['source_sha256'];check_hashes(sources,'source closure')
    need(all(x.startswith('src/') for x in sources),'Non-source frozen closure')
    names.update(sources)
    check_hashes(req['input_sha256'],'frozen inputs')
    check_hashes(req['protected_before'],'protected original submissions/media')
    merged={**sources,req['plan']['path']:req['plan']['sha256']}
    need(pre['status']=='passed' and pre['source_sha256']==merged,'Independent preflight source/plan incomplete')
    pre_inputs=pre['input_sha256_independently_reverified'] if number==114 else pre['input_sha256']
    additional={paths['preflight_review']:reference(paths['preflight_review'])['sha256']}
    if number==114:
        for path in (group+'/R114_driver_selfcheck_execution.json',group+'/R114_trainer_selfcheck_execution.json',
                     'outputs/round97/evaluation/results.json'):
            additional[path]=reference(path)['sha256']
    need(isinstance(pre_inputs,dict) and pre_inputs
         and req['input_sha256']=={**pre_inputs,**additional},
         'Frozen inputs are not exactly preflight inputs plus review')
    require_common_review(post,number)
    need(post['source_sha256']==merged and post['plan']==req['plan']
         and post['protected_unchanged'] is True and post['selected_submission']=='v22',
         'Postfit source/plan/protection/selection differs')
    for key in ('request','seal','results','training_execution','evaluation_execution','preflight_review'):
        need(post[key]==reference(paths[key]),'Postfit exact '+key+' differs')
    collect_record(post['review_tool'],names)
    check_hashes(post['review_dependency_source_sha256'],'postfit dependency sources')
    names.update(post['review_dependency_source_sha256'])
    postfit_execution(number,post,names,postfit_overrides)
    need(seal['request']==reference(paths['request']) and seal['origin_order']==list(STAGES)
         and seal['new_fits']==4 and seal['query_scores_computed'] is False
         and set(seal['records'])==set(STAGES),'Four-model seal incomplete')
    need(result['status']=='four_reused_fixed_origins_evaluated' and result['round']==number
         and result['origin_order']==list(STAGES) and set(result['stages'])==set(STAGES)
         and result['source_sha256']==sources and result['plan']==req['plan']
         and result['request']==reference(paths['request']) and result['seal']==reference(paths['seal'])
         and result['new_fits']==4 and result['new_CSV'] is False and result['full_data_fit'] is False
         and result['Public_score'] is None and result['Private_score'] is None
         and result['selected_submission']=='v22','Four-origin evaluation incomplete')
    check_combined(result['gates'])
    need(post['manual_gate_verdicts']==manual_gates(result['gates'],
         result['stages']['offset_2023_02_11']['both_point_metrics_improve']),
         'Independent manual gates differ from actual four-origin results')
    train=read(paths['training_execution']);evaluation=read(paths['evaluation_execution'])
    need(train['exit_code']==evaluation['exit_code']==0 and train['failure'] is None
         and evaluation['failure'] is None and train['automatic_retry'] is False
         and train['request']==evaluation['request']==reference(paths['request'])
         and evaluation['seal']==reference(paths['seal'])
         and evaluation['training_execution']==reference(paths['training_execution'])
         and evaluation['training_already_finished_before_scoring'] is True
         and evaluation['query_labels_used_for_scoring_only'] is True
         and evaluation['new_fits']==0 and evaluation['new_CSV'] is False,
         'Training/evaluation completion or separation unverified')
    for name,flag in [('freeze_execution','--freeze'),('train_execution','--train'),('evaluate_execution','--evaluate')]:
        check_execution(group+'/'+name+'.json',f'src/round{number}_research.py',flag,names)
    need(set(post['artifacts'])==set(post['trainmeta'])==set(STAGES),'Postfit missing/extra origins')
    for origin in STAGES:
        p=folder+'/'+origin; stage_name=p+'/stage.json';stage=read(stage_name)
        stage_ref=reference(stage_name);meta_name=stage['model_metadata']['path'];meta=read(meta_name)
        names.update((stage_name,meta_name,p+'/evaluation/results.json'))
        need(seal['records'][origin]==stage_ref and stage['stage']==origin
             and stage.get('candidate_round',stage.get('round'))==number
             and stage['query_labels_available'] is False and stage['query_scores_computed'] is False
             and stage['context_before']==stage['context_after'],'Stage identity/context changed')
        need(stage['model_metadata']==reference(meta_name)
             and read(p+'/evaluation/results.json')==result['stages'][origin], 'Stage metadata/evaluation differs')
        audited=post['artifacts'][origin]
        need(audited['stage']==stage_ref and audited['training_metadata']==reference(meta_name)
             and audited['evaluation']==reference(p+'/evaluation/results.json')
             and audited['sealed_models']==stage['sealed_models']
             and post['trainmeta'][origin]==meta, 'Exact stage/model/postfit metadata differs')
        sealed_model_set(number,p,stage['sealed_models'])
        for record in stage['sealed_models']:
            need(str(PurePosixPath(record['path']).parent)==p+'/model','Model path outside own origin')
            check_opaque_record(record)
        check_model_metadata(number,origin,stage,meta,req)
        need(set(stage['layers'])=={'native','base','reference','spatial','median'},'Missing five layers')
        for record in stage['layers'].values():check_identity(record)
        for key in ('raw_signed_total','raw_unit_prediction'):
            if key in stage:
                need(audited[key]==stage[key],'Postfit raw identity differs');check_identity(stage[key])
        if number==117:
            need(req['new_prior_builds']==seal['new_prior_builds']==result['new_prior_builds']==4
                 and evaluation['new_prior_builds']==0,'R117 four prior builds incomplete')
            need(audited['sealed_prior_artifacts']==stage['sealed_prior_artifacts']
                 and audited['prior_preparation']==stage['prior_preparation'], 'Private prior exact seal differs')
            for record in stage['sealed_prior_artifacts']:check_opaque_record(record)
            check_opaque_record(stage['prior_preparation'])
            old=stage['original_prepared_array_identities'];new=stage['prepared_array_identities']
            need(set(old)==set(new) and len(old)==9 and all(old[k]==new[k] for k in old
                 if k!='train_price_factor_million'),'R117 changed more than fit prior factor')
        names.add(p+'/train.log')
        names.add(p+'/model/'+('started.json' if number==116 else 'run.json'))
    if number in (114,116):interpretation_binding(number,names)
    return check_disposition(number,disposition,result)


def pure_X_round(names,postfit_overrides):
    directory='outputs/round115';group='outputs/research_round115_20261008'
    paths={k:directory+'/'+k+'.json' for k in ('request','started','results')}
    paths.update({k:group+'/'+k+'.json' for k in ('preflight_review','postfit_review')})
    names.update(paths.values())
    req,started,result,pre,post=(read(paths[k]) for k in ('request','started','results','preflight_review','postfit_review'))
    require_common_review(post,115)
    sources=req['source_sha256'];check_hashes(sources,'R115 source closure')
    merged={**sources,req['plan']['path']:req['plan']['sha256']}
    need(pre['status']=='passed' and pre['source_sha256']==merged
         and post['source_sha256']==merged and result['source_sha256']==sources,
         'Pure-X source closure differs')
    check_hashes(req['input_sha256'],'R115 typed inputs')
    need(req['input_sha256']==result['input_sha256']==pre['input_sha256'],
         'Pure-X typed readset differs')
    collect_record(req['plan'],names);names.update(sources)
    need(started['request']==reference(paths['request']) and result['request']==reference(paths['request'])
         and result['independent_preflight']==req['independent_preflight']==reference(paths['preflight_review'])
         and result['plan']==req['plan']==post['plan']
         and result['status']=='all_four_origins_pure_X_audit_completed'
         and result['round']==115 and set(result['stages'])==set(STAGES), 'Pure-X completion differs')
    for key in ('request','started','results','preflight_review'):
        need(post[key]==reference(paths[key]),'Pure-X postfit binding differs: '+key)
    need(result['new_fits']==result['GPU_calls']==result['MAE_calls']==result['bootstrap_calls']
         ==result['model_loads']==result['price_prior_builds']==0
         and result['NPZ_opened'] is False and result['query_labels_read'] is False
         and result['raw_price_pickle_opened'] is False and result['new_CSV'] is False
         and result['inputs_and_sources_unchanged'] is True
         and result['automatic_model_experiment_authorized'] is False
         and result['Public_score'] is None and result['Private_score'] is None,
         'Pure-X audit includes forbidden fitting/scoring/labels')
    need(post['aggregate_privacy_numeric_only_statistics_verified'] is True
         and post['source_and_input_before_after_verified'] is True
         and post['prices_or_query_y_read_by_reviewer'] is False
         and post['housing_X_or_arrays_deserialized_by_reviewer'] is False
         and post['independent_full_X_statistics_recomputed'] is False
         and post['artifacts']==result['stages'], 'Pure-X audit scope/identity differs')
    for key in ('review_tool','formal_author_selfcheck','root_independent_arithmetic','freeze_execution','run_execution'):
        collect_record(post[key],names)
    postfit_execution(115,post,names,postfit_overrides)
    check_execution(group+'/freeze_execution.json','src/round115_prior_support_x_audit.py','--freeze',names)
    check_execution(group+'/run_execution.json','src/round115_prior_support_x_audit.py','--run',names)
    for origin in STAGES:
        record=result['stages'][origin];collect_record(record,names)
        agg=read(record['path'])
        need(record['path']==directory+'/'+origin+'/aggregates.json'
             and agg['status']=='pure_X_support_exposure_completed'
             and agg['round']==115 and agg['stage']==origin and agg['request']==reference(paths['request']),
             'Pure-X aggregate authority differs')


def check_sidecars(names):
    """Only exact siblings, never discover entire output directories."""
    additions=set()
    for name in tuple(names):
        if name.endswith('.sha256'):continue
        path=PurePosixPath(name)
        for sibling in (str(path.with_suffix('.sha256')),name+'.sha256'):
            if safe(ROOT,sibling).is_file():
                check_sidecar(sibling,safe(ROOT,sibling).read_bytes());additions.add(sibling)
    names.update(additions)


def bounded_extra(name,include_feasibility=False):
    path=policy(name)
    if path.parts[0]=='src':
        need(name in SOURCE_FILES or re.fullmatch(r'src/round118_[A-Za-z0-9_]+\.py',name),
             'Extra source must be in fixed114--118 scope')
    if '118' in name:
        need(include_feasibility and (name=='reports/round118_feasibility_plan.md'
             or str(path.parent)==FEASIBILITY_GROUP),
             'R118 requires explicit inclusion and sealed completed feasibility inventory')
    if path.parts[0]=='outputs' and path.parts[1].startswith('round'):
        need(path.name in {'request.json','seal.json','results.json','stage.json','aggregates.json',
             'training.json','metadata.json','run.json','started.json','train.log'} or path.suffix=='.sha256',
             'Extra round artifact is not an approved aggregate evidence type')
    return name


def feasibility_inventory(names):
    group=FEASIBILITY_GROUP;manifest=read(group+'/manifest.json')
    expected={group+'/'+x for x in FEASIBILITY_ARTIFACTS}
    need(manifest['status']=='sealed_bounded_feasibility_inventory'
         and set(manifest['artifacts'])==expected and manifest['author_checks_not_independent_review'] is True
         and manifest['independent_review_pending_root'] is True
         and manifest['no_existing_project_source_or_artifact_mutated'] is True
         and manifest['no_training_or_CSV_or_Kaggle_changes'] is True
         and manifest['author_synthetic_check_count']==30,'R118 sealed bounded author inventory incomplete')
    check_hashes(manifest['artifacts'],'R118 sealed aggregate/source artifacts')
    result=read(group+'/feasibility.json');readset=read(group+'/readset.json')
    checks=read(group+'/author_synthetic_checks.json');limits=read(group+'/final_design_limits.json')
    need(readset['all_unchanged'] is True and len(readset['input_sha256'])==manifest['original_inputs_unique_count']==77,
         'R118 complete readset differs')
    check_hashes(readset['input_sha256'],'R118 readset, opaque hashes only')
    collect_record(result['plan'],names)
    need(result['plan']['path']=='reports/round118_feasibility_plan.md'
         and result['new_fits']==result['GPU_fits']==result['models_loaded']==0
         and result['price_values_loaded']==result['query_y_loaded']==0
         and result['prediction_values_loaded']==result['private_states_loaded']==0
         and result['raw_price_pickle_opened'] is False and result['NPZ_opened'] is False
         and result['valid_test_read'] is False and result['MAE_computed'] is False
         and result['bootstrap_computed'] is False and result['new_CSV'] is False
         and result['Public_score'] is None and result['Private_score'] is None
         and result['input_files_unchanged'] is True,'R118 must remain date/index/metadata-only feasibility')
    need(checks['status']=='passed' and checks['author_review_not_independent'] is True
         and checks['check_count']==len(checks['checks'])==30
         and checks['new_actual_data_deserializations_by_selfcheck']==0
         and checks['source_sha256']==manifest['artifacts'][group+'/seal_evidence.py'],
         'R118 author checks are not exact bounded checks')
    need(limits['status']=='bounded_inventory_complete_no_model_or_cutoff_selected'
         and limits['original_inputs_unique_count']==77 and limits['new_fits']==limits['GPU_fits']==0
         and limits['new_CSV'] is False and limits['new_scores_or_bootstrap'] is False
         and limits['formal_constructibility_not_strong_validation'] is True
         and limits['all_new_prefix_models_encoders_priors_reference_spatial_still_required'] is True
         and limits['geo_cache_rowwise_source_and_metadata_supported_only_not_binary_replayed'] is True,
         'R118 feasibility must preserve actual rebuild/time/geo limitations')
    names.update(expected|{group+'/manifest.json'})


def inventory(dispositions,extras,postfit_overrides,include_feasibility):
    check_record(IMPLEMENTATION_PARENT)
    names=set(SOURCE_FILES)|set(REPORT_FILES)|{'outputs/research_continue_20261008/user_request.json'}
    statuses={n:model_round(n,dispositions[n],names,postfit_overrides) for n in ROUNDS}
    pure_X_round(names,postfit_overrides)
    if include_feasibility:feasibility_inventory(names)
    # Individually named author evidence and command proofs are required, not
    # inferred from directory contents or copied wholesale with private state.
    formal={114:('R114_driver_selfcheck','R114_trainer_selfcheck'),
            116:('R116_driver_selfcheck','R116_helper_selfcheck'),
            117:('R117_driver_selfcheck','R117_prior_selfcheck','R117_trainer_selfcheck')}
    for n,stems in formal.items():
        group=f'outputs/research_round{n}_20261008'
        for stem in stems:
            name=group+'/'+stem+'_execution.json'
            need(read(name)['exit_code']==0,'Required formal CPU selfcheck failed/missing')
            names.update((name,group+'/'+stem+'.log'))
        for leaf in ('launch.py','freeze.log','train.log','evaluate.log'):
            names.add(group+'/'+leaf)
    for leaf in ('launch.py','run.log','freeze.log','formal_author_selfcheck.log'):
        names.add('outputs/research_round115_20261008/'+leaf)
    author=(
        'direction_review_a/R114_driver_author_selfcheck.json',
        'direction_review_b/R114_author_selfcheck.json',
        'direction_review_b/R115_author_selfcheck.json',
        'direction_review_a/R116_helper_author_selfcheck.json',
        'direction_review_a/R116_driver_author_selfcheck.json',
        'direction_review_b/R117_author_selfcheck.json',
        'direction_review_a/R117_driver_author_selfcheck.json',
        'direction_review_a/package_four_origins_author_selfcheck.json',
        'direction_review_a/package_four_origins_author_review.md')
    names.update('outputs/research_continue_20261008/'+x for x in author)
    for name in extras:names.add(bounded_extra(name,include_feasibility))
    missing=[x for x in sorted(names) if not safe(ROOT,x).is_file()]
    need(not missing,'Missing completion evidence; parent must resolve actual status:\n'+'\n'.join(missing))
    check_sidecars(names)
    entries=[]
    for name in sorted(names):
        payload=safe(ROOT,name).read_bytes();policy(name,payload)
        if name.endswith('.sha256'):check_sidecar(name,payload)
        entries.append({'source':name,'destination':destination(name),'bytes':len(payload),'sha256':sha_bytes(payload)})
    need(len({x['destination'] for x in entries})==len(entries),'Duplicate copy destination')
    return entries,statuses


def summary(dispositions,statuses,include_feasibility):
    lines=['# 2026-10-08 四起點後續研究：R114–R117','',
        'v22仍為使用者選定提交；已回報Public MAE為1,998,568.5706元，Private未知。新研究／v25探索候選不自動改Kaggle勾選。',
        '本批與先前R111／R112六fit＋R113三fit分開。本批共12 native fits：R114四NN＋R116四CatBoost＋R117四NN；R117另四次fit-only先驗重建。R115是四起點純X診斷，沒有fit或MAE。',
        '以下是重用歷史窗口的離線TWD MAE差；負值較保存v22好，新候選Public／Private均未知。','',
        '|固定方法／理由|dev ΔMAE|latest ΔMAE|aux ΔMAE|Feb ΔMAE|探索／強combined|真實終態|',
        '|---|---:|---:|---:|---:|---|---|']
    for n in ROUNDS:
        result=read(f'outputs/round{n}/results.json');g=result['gates']
        d=[result['stages'][s]['metrics']['T']['overall_mae']-result['stages'][s]['metrics']['v22']['overall_mae'] for s in STAGES]
        need(all(math.isfinite(x) for x in d),'Nonfinite summary')
        status=statuses[n];reason=next(iter(dispositions[n].values())).replace('|','／').replace('\n',' ')
        role=('探索通過但全量發行未完成：' if status=='eligible_full_release_deferred' else 'combined門檻失敗，不產CSV：')+reason
        lines.append(f'|[R{n}](../evidence/outputs/round{n}/results.json)：{METHODS[n]}|'+
            '|'.join(f'{x:+,.2f}' for x in d)+f"|{g['combined_exploratory_submission']['passed']}／{g['combined_promotion']['passed']}|{role}|")
    lines+=['','原三August窗與固定Feb起點分開評估，兩套combined均要求對應三窗gate AND 額外origin gate。探索容許每窗至多0.25%退步且限制大群／季度退步；強gate另要求全面改善與最新至少0.5%。不因結果更改門檻。',
        'R115支持／ESS與同案暴露的診斷只提供一次受限fold比較的理由，不是價格改善證據。R117交易fold允許同案peer，因此即使OOF排除自身，也不是group隔離或chronological OOF；query仍使用原全fit先驗。',
        'R114／R116原診斷內false只描述封存時的預測投影／count統計；事後raw／unprojected MAE實際使用query真價，僅於四origin全部封存及成功退出後。另附明確語意binding，不改原source、results、gates或舊收據。','',
        '|研究|真命令訓練／run秒數|另評分秒數|獨立審查秒數來源|','|---|---:|---:|---|']
    for n in (114,115,116,117):
        group=f'outputs/research_round{n}_20261008'
        train=read(group+('/run_execution.json' if n==115 else '/train_execution.json'))
        ev='無MAE' if n==115 else f"{read(group+'/evaluate_execution.json')['elapsed_seconds']:.3f}"
        post=read(group+'/postfit_review.json');secs=post.get('review_elapsed_seconds')
        measured=f'{secs:.3f}' if isinstance(secs,(int,float)) else '收據未提供，不估造'
        lines.append(f'|R{n}|{train["elapsed_seconds"]:.3f}|{ev}|[postfit]('+f'../evidence/{group}/postfit_review.json'+f') {measured}|')
    lines+=['','沒有把wrapper秒數、driver秒數或各fit相加成新的ETA。','']
    if include_feasibility:
        lines+=['[R118可行性盤點](../evidence/outputs/research_round118_feasibility_20261008/feasibility.md)：77個精確inputs、30作者人工checks。只讀日期／索引／metadata及保存向量opaque SHA；零fit／MAE／價格或逐筆預測載入。作者檢查不冒稱獨立結果審查。',
            '既有早期地圖時間邊界可支持2021-01-03起點（prefix126,957／query7,848），但完整forward block仍須另建4樹＋2NN及各prefix encoder／prior／reference／spatial。可建構不等於強驗證；這份盤點沒有選融合模型或自動訓練，之後工作須另立規約。','']
    lines+=[
        '限制：反覆使用同窗有選型偏差，bootstrap不是Private勝率，GPU seed不保證逐bytes重訓。保存重載與來源完整，不等於新電腦端到端重訓已完成。未去內政部成交系統對答案。',
        '程式／證據包排除課程原表、CSV、逐筆價格／向量、模型、encoder／prior／reference狀態。僅有本機不公開產物的SHA引用。既有379歷史、13production與所有舊媒體／收據保持。',
        f'[本批複製manifest](../{COPY_MANIFEST})；本工具不Git操作、不上傳／勾選Kaggle，不使用額外點數或重置。','']
    payload='\n'.join(lines).encode('utf-8');check_text(payload,SUMMARY);return payload


def run(dispositions,extras,apply,postfit_overrides,include_feasibility):
    check_git_base()
    current=package_files();template,protected=protect_existing(current)
    need(len(protected)==3226,'All3226 old non-navigation files must be protected')
    entries,statuses=inventory(dispositions,extras,postfit_overrides,include_feasibility)
    document=summary(dispositions,statuses,include_feasibility)
    archives=[SNAPSHOT+'/before/'+x for x in ('SOURCE_MANIFEST.json',)+NAVIGATION]
    planned=[x['destination'] for x in entries]+[SUMMARY,COPY_MANIFEST]+archives
    for name in planned:safe(REPO,name)
    for name in [SUMMARY,COPY_MANIFEST]+archives:
        need(not safe(REPO,name).exists(),'Preserve previous/partial package: '+name)
    for item in entries:
        target=safe(REPO,item['destination'])
        if target.exists():
            need(item['destination'] in current and sha(target)==item['sha256'],
                 'Existing/unlisted destination differs; no overwriting')
    planned_receipt={'status':'ready_to_copy' if apply else 'preflight_passed_no_writes',
        'base_HEAD':BASE_HEAD,'source_manifest_count':BASE_MANIFEST_COUNT,
        'selected_count':len(entries),'selected_bytes':sum(x['bytes'] for x in entries),
        'round_dispositions':dispositions,'computed_round_status':statuses,
        'R118_sealed_author_feasibility_included':include_feasibility,
        'new_model_fits':0,'git_or_network':False,'private_artifacts_copied':False}
    if not apply:return planned_receipt
    # Recheck old member protection before the first write; mutation is explicit.
    verify_protected(protected)
    for item in entries:
        payload=safe(ROOT,item['source']).read_bytes()
        need(sha_bytes(payload)==item['sha256'],'Source changed after preflight')
        target=safe(REPO,item['destination'])
        if not target.exists():write_new(target,payload)
        need(target.read_bytes()==payload,'Copied bytes differ')
    write_new(safe(REPO,SUMMARY),document)
    generated=[{'path':SUMMARY,'sha256':sha_bytes(document),'bytes':len(document)}]
    for name in ('SOURCE_MANIFEST.json',)+NAVIGATION:
        payload=safe(REPO,name).read_bytes();archive=SNAPSHOT+'/before/'+name
        write_new(safe(REPO,archive),payload)
        generated.append({'path':archive,'sha256':sha_bytes(payload),'bytes':len(payload)})
    verify_protected(protected)
    receipt={**planned_receipt,'status':'copied_exact_bytes_local_only',
        'created_at_utc':datetime.now(timezone.utc).isoformat(),'files':entries,'generated':generated,
        'copied_count':sum(x['destination'] not in current for x in entries),
        'verified_equal_existing_count':sum(x['destination'] in current for x in entries),
        'protected_before':protected,'historical_379_manifest_unchanged':True,
        'production_13_sources_unchanged':True,'all_other_preexisting_package_files_unchanged':True,
        'former_3014_protected_included':True,'navigation_finish_separate':True,
        'clean_environment_end_to_end_retraining_claimed':False,
        'excluded':['raw course data','CSV','rows','model/array/private learned states','credentials']}
    write_new(safe(REPO,COPY_MANIFEST),encoded(receipt))
    additions=[x['destination'] for x in entries]+[x['path'] for x in generated]+[COPY_MANIFEST]
    refreshed=refresh_top(template,additions);verify_protected(protected)
    return {**planned_receipt,'status':receipt['status'],'copy_manifest':COPY_MANIFEST,
        'copy_manifest_sha256':sha(safe(REPO,COPY_MANIFEST)),'SOURCE_MANIFEST':refreshed,
        'finish':'Parent may add dated navigation notes preserving old bodies, then --refresh-manifest; publication separate.'}


def self_check():
    """Synthetic scalar/hash/schema guards only; no workspace evidence readers."""
    import copy
    checks=[]
    def ok(name,condition):
        need(condition,'CPU selfcheck failed: '+name);checks.append(name)
    def reject(name,action):
        try:action()
        except (ValueError,KeyError,TypeError,UnicodeError):checks.append(name);return
        raise ValueError('CPU selfcheck accepted malformed evidence: '+name)
    good={'status':'passed','round':117,'new_model_fits':0,'GPU_work':False,
          'Private_score':None,'Public_score':None,'new_CSV':False,'blocking_findings_remaining':[]}
    require_common_review(good,117);ok('complete_common_review',True)
    for key,bad in [('status','pending'),('round',116),('new_model_fits',1),('GPU_work',True),
        ('Private_score',1.0),('Public_score',1.0),('new_CSV',True),('blocking_findings_remaining',['unresolved'])]:
        value=copy.deepcopy(good);value[key]=bad
        reject('common_'+key,lambda value=value:require_common_review(value,117))
    for key in good:
        value=copy.deepcopy(good);value.pop(key)
        reject('missing_review_'+key,lambda value=value:require_common_review(value,117))
    for a in (False,True):
        for e in (False,True):
            for strong in (False,True):
                gate={'august':{'exploratory_submission':{'components':{'rule':a},'passed':a},
                    'promotion':{'components':{'rule':strong},'adoption_research_gate_passed':strong,
                        'confirmation_fractional_improvement':0.0}},
                    'extra_origin':{'components':{'rule':e},'passed':e},
                    'combined_exploratory_submission':{'components':{'original_three_windows':a,'fixed_extra_origin':e},'passed':a and e},
                    'combined_promotion':{'components':{'original_three_windows':strong,'fixed_extra_origin':e},'passed':strong and e}}
                check_combined(gate);ok(f'combined_{a}_{e}_{strong}',True)
                expected=manual_gates(gate,False)
                ok(f'manual_{a}_{e}_{strong}',expected['extra_origin']['both_point_metrics_improve'] is False)
                broken=copy.deepcopy(gate);broken['combined_exploratory_submission']['passed']=not(a and e)
                reject(f'combined_not_AND_{a}_{e}_{strong}',lambda broken=broken:check_combined(broken))
    for name in ['../secret.json','C:/else.json','outputs\\round117\\results.json','outputs/round117/../results.json',
        'outputs//round117/results.json','outputs/node_modules/x.json','outputs/round117/model/model.pt',
        'outputs/round117/prior/prior_evidence.json','outputs/round117/prior/anything.json',
        'outputs/round117/model/feature_state.json','outputs/round117/prepared.npz',
        'outputs/round117/model/predictions_twd.npy','outputs/round117/submission.csv','reports/round119_plan.md',
        'outputs/research_round999_20261008/results.json','outputs/round117/test_rows.json']:
        reject('path_'+name,lambda name=name:policy(name))
    for name in ['src/round117_research.py','reports/round117_plan.md','outputs/round117/results.json',
        'outputs/research_round117_20261008/postfit_review.json']:
        ok('allowed_'+name,policy(name).as_posix()==name)
    for key,value in [('rows',[[1,2]]),('predictions',[1,2]),('targets',[1]),('state_dict',{}),
        ('cells',{'learned':1}),('preview',{'row':1}),('arbitrary',list(range(256))),
        ('arbitrary',{'編號':1,'總價元':2})]:
        reject('payload_'+key+str(type(value)),lambda key=key,value=value:check_aggregate({key:value}))
    for payload in [b'{"a":1,"a":2}',b'{"a":NaN}',b'{"a":Infinity}']:
        reject('invalid_JSON_'+payload.decode(),lambda payload=payload:json_value(payload))
    check_aggregate({'metrics':{'overall_mae':3.0},'array':{'shape':[20,144],'values_sha256':'a'*64}})
    ok('aggregate_shapes_and_metrics_allowed',True)
    secret=''.join(['github','_pat_'])+'x'*30
    reject('credential_copy_guard',lambda:policy('outputs/round117/results.json',encoded({'token':secret})))
    reject('credential_generated_guard',lambda:check_text(secret.encode(),'generated'))
    reject('CSV_log_guard',lambda:policy('outputs/round117/train.log',b'1, -12345\n'))
    reject('row_JSON_log_guard',lambda:policy('outputs/round117/train.log',b'{"rows":[[1,2]]}\n'))
    original=b'original body\n'
    for value in [b'2026-10-08 note\n'+original,original+b'2026-10-08 note\n']:
        check_navigation_note(original,value,'README.md');ok('preserve_body_'+str(len(checks)),True)
    reject('navigation_rewrite',lambda:check_navigation_note(original,b'2026-10-08 replacement','README.md'))
    reject('navigation_undated',lambda:check_navigation_note(original,original+b'new','README.md'))
    reasons=['114=combined fixed thresholds failed','116=all fixed gates failed','117=combined fixed thresholds failed']
    ok('explicit_three_dispositions',set(parse_dispositions(reasons,[]))==set(ROUNDS))
    reject('missing_round_disposition',lambda:parse_dispositions(reasons[:2],[]))
    reject('duplicate_round_disposition',lambda:parse_dispositions(reasons+[reasons[0]],[]))
    reject('short_reason',lambda:parse_dispositions(['114=no']+reasons[1:],[]))
    reject('other_round_disposition',lambda:parse_dispositions(['118=not in this model batch']+reasons[1:],[]))
    reject('R118_not_yet_completed',lambda:bounded_extra('reports/round118_feasibility_plan.md'))
    item={'fit_rows':4,'query_rows':3,'prepared_array_identities':{'test':'synthetic identity'}}
    req={'recipe':{'configuration':{'seed':42}}}
    base={'status':'completed','saved_model_reload_exact':True,'query_labels_used':False,
        'fit_rows':4,'query_rows':3,'reload_rows':3,'configuration':{'seed':42},
        'epochs':[{'epoch':i} for i in range(1,31)]}
    signed={**base,'saved_raw_total_reload_exact':True,'training_total_clipped':False}
    check_model_metadata(114,'development',item,signed,req);ok('signed_NN_contract',True)
    ratio={**base,'saved_ratio_reload_exact':True,'actual_model_class':'round47_nn_train.EmbeddingMLP',
        'changed_configuration_fields':['train_price_factor_construction'],'ML_configuration_changes':[],
        'loss_configuration_changes':[],'input_arrays':item['prepared_array_identities']}
    check_model_metadata(117,'development',item,ratio,req);ok('transaction_ratio_contract',True)
    depth={**base,'raw_and_TWD_reload_exact':True,'actual_grow_policy':'Depthwise','actual_boosting_type':'Plain',
        'actual_task_type':'GPU','actual_ctr_target_border_count':1,'actual_max_ctr_complexity':2,
        'actual_trees':2999,'actual_model_depth':8,'actual_seed':42,'actual_min_data_in_leaf':1,
        'tree_leaf_counts_count':2999,'tree_leaf_counts_distribution':{'3':2999}}
    check_model_metadata(116,'development',item,depth,req);ok('Depthwise_nonpower2_valid',True)
    for n,template,key,bad in [(114,signed,'training_total_clipped',True),
        (114,signed,'saved_raw_total_reload_exact',False),(117,ratio,'saved_ratio_reload_exact',False),
        (117,ratio,'actual_model_class','some.new.Model'),(117,ratio,'changed_configuration_fields',['other']),
        (117,ratio,'ML_configuration_changes',['depth']),(116,depth,'actual_boosting_type','Ordered'),
        (116,depth,'actual_ctr_target_border_count',8),(116,depth,'actual_trees',2998),
        (116,depth,'actual_grow_policy','SymmetricTree'),(116,depth,'tree_leaf_counts_distribution',{'257':2999}),
        (116,depth,'tree_leaf_counts_distribution',{'3':2998}), (116,depth,'raw_and_TWD_reload_exact',False)]:
        value=copy.deepcopy(template);value[key]=bad
        reject(f'metadata_{n}_{key}_{len(checks)}',lambda n=n,value=value:check_model_metadata(n,'development',item,value,req))
    prefix='outputs/round117/development'
    records=[{'path':prefix+'/model/'+name,'sha256':'a'*64}
             for name in ('model.pt','run.json','training.json','ratio_predictions.npy','predictions_twd.npy')]
    sealed_model_set(117,prefix,records);ok('exact_five_model_artifacts',True)
    reject('missing_model_artifact',lambda:sealed_model_set(117,prefix,records[:-1]))
    reject('extra_model_artifact',lambda:sealed_model_set(117,prefix,records+[{'path':prefix+'/model/other.json','sha256':'a'*64}]))
    reject('duplicate_model_artifact',lambda:sealed_model_set(117,prefix,records[:-1]+[records[0]]))
    wrong=copy.deepcopy(records);wrong[0]['path']='outputs/round117/confirmation/model/model.pt'
    reject('wrong_origin_model_artifact',lambda:sealed_model_set(117,prefix,wrong))
    wrong=copy.deepcopy(records);wrong[0]['size']=123
    reject('nonexact_artifact_record',lambda:sealed_model_set(117,prefix,wrong))
    ok('execution_override_own_round',parse_postfit_executions([
        '117=outputs/research_round117_20261008/independent_postfit_execution_v2.json'])=={
            117:'outputs/research_round117_20261008/independent_postfit_execution_v2.json'})
    reject('execution_override_other_round',lambda:parse_postfit_executions([
        '117=outputs/research_round116_20261008/independent_postfit_execution_v2.json']))
    reject('execution_override_arbitrary_JSON',lambda:parse_postfit_executions([
        '117=outputs/research_round117_20261008/preflight_review.json']))
    old_read=globals()['read']
    try:
        command={'command':['C:/NYCU_ML/.venv/Scripts/python.exe','-B','-X','utf8',
            'C:/NYCU_ML/src/round117_research.py','--train'],'elapsed_seconds':2.0,'exit_code':0}
        globals()['read']=lambda _:command
        check_execution('synthetic','src/round117_research.py','--train',set());ok('formal_command_entry_flag',True)
        for key,bad in [('exit_code',1),('elapsed_seconds',-1),('command',['wrong.py','--train'])]:
            command[key],original_value=bad,command[key]
            reject('formal_command_'+key,lambda:check_execution('synthetic','src/round117_research.py','--train',set()))
            command[key]=original_value
        reject('formal_wrong_flag',lambda:check_execution('synthetic','src/round117_research.py','--evaluate',set()))
    finally:globals()['read']=old_read
    need(len(checks)==len(set(checks)),'Selfcheck labels must be unique')
    return {'status':'passed','checks':checks,'check_count':len(checks),
        'scope':'synthetic CPU guards; no actual package check or copied files',
        'new_model_fits':0,'GPU_calls':0,'query_prices_read':False,'arrays_models_loaded':False,
        'package_check_run':False,'package_copy_run':False,'git_or_network':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check',action='store_true')
    mode.add_argument('--apply',action='store_true')
    mode.add_argument('--refresh-manifest',action='store_true')
    mode.add_argument('--self-check',action='store_true')
    parser.add_argument('--no-release',action='append',default=[],metavar='ROUND=REASON')
    parser.add_argument('--defer-release',action='append',default=[],metavar='ROUND=REASON')
    parser.add_argument('--extra-file',action='append',default=[],metavar='EXACT_PATH')
    parser.add_argument('--postfit-execution',action='append',default=[],metavar='ROUND=EXACT_PATH')
    parser.add_argument('--include-feasibility',action='store_true',help='Include completed sealed R118 author inventory; no model/gate')
    args=parser.parse_args()
    if args.self_check:
        need(not args.no_release and not args.defer_release and not args.extra_file
             and not args.postfit_execution and not args.include_feasibility,'Selfcheck accepts no copy inputs')
        result=self_check()
    elif args.refresh_manifest:
        need(not args.no_release and not args.defer_release and not args.extra_file
             and not args.postfit_execution and not args.include_feasibility,'Finish accepts no new evidence')
        result=finish_manifest()
    else:
        result=run(parse_dispositions(args.no_release,args.defer_release),args.extra_file,args.apply,
                   parse_postfit_executions(args.postfit_execution),args.include_feasibility)
    print(json.dumps(result,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False))


if __name__=='__main__':main()
