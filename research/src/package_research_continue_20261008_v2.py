"""V2 bounded R111/R112 plus fixed-origin R113 evidence package; stdlib only.

--check is read-only and fails until formal evaluation plus independent preflight
and postfit pass. --apply is for the parent after source review; no git/network/ML
imports or training. Approved base HEAD59e646d,3017manifest members/3018tracked,
379history/13production and all old receipts are preserved. Only exact named
aggregate evidence is copied; private models/arrays/course rows/CSV are excluded.

Supply one --release/--no-release/--defer-release disposition per111/112. A passed
release requires exact-bound independent final review; failed gates require
no-release with absence of v28/v29 CSV. An eligible but unfinished fullfit must be
explicitly deferred. --extra-file names additional bounded source/report/aggregate
files individually. It is never a broad directory import.

Navigation finish is separate: dated prefix/suffix retaining complete original
README/research README/HANDOFF bodies, then --refresh-manifest. No upload or Kaggle
selection is performed; Private remains unknown. A failed partial copy is left
for inspection and cannot be silently rerun or described as atomic.
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
DATE = '20261008'
ROUNDS = (111, 112)
VERSIONS = {111: 28, 112: 29}
STAGES = ('development', 'confirmation', 'auxiliary')
GROUP = 'outputs/research_round111_112_20261008'
CONTINUE_GROUP = 'outputs/research_continue_20261008'
EXTRA_GROUP = 'outputs/research_round113_20261008'
EXTRA_GATE = EXTRA_GROUP + '/round113_release_gate.json'
EXTRA_REVIEW = EXTRA_GROUP + '/postfit_review.json'
EXTRA_PLAN = 'reports/round113_plan.md'
EXTRA_PLAN_SHA256 = 'a21116136190aadf94d89329dd1b9dd98afb62dd493ef6863b356f50ea01dc2c'
EXTRA_RESULTS = 'outputs/round113/results.json'
EXACT_BUILDER = 'outputs/research_resume_20261008/build_submission.mjs'
GROUPS = (GROUP, CONTINUE_GROUP, EXTRA_GROUP)
SNAPSHOT = 'evidence/outputs/research_package_continue_20261008'
COPY_MANIFEST = SNAPSHOT + '/copy_manifest.json'
SUMMARY = 'docs/RESEARCH_CONTINUE_20261008.md'
NAVIGATION = ('README.md', 'research/README.md', 'HANDOFF.md')
BASE_HEAD = '59e646d77b7581cfd8abf229a1f62d26606ca1af'
BASE_MANIFEST_SHA256 = '046a7ede2c92b0278b981af581ccebf13dae3619360a9720c0e1b2b0248309cf'
BASE_MANIFEST_COUNT = 3017
BASE_TRACKED_COUNT = 3018
ALLOWED_SUFFIXES = {'.py', '.mjs', '.md', '.json', '.sha256', '.log'}
FORBIDDEN_PARTS = {'node_modules', '__pycache__', '.git', '.venv', 'venv',
                   'research_csv_intermediate_20261008', 'rawinput', 'raw_input', 'rawinputs', 'frames',
                   'private', 'runs'}
FORBIDDEN_NAMES = {'feature_state.json', 'state.json', 'state_summary.json', 'price_prior.json',
                   'prior_state.json', 'encoder_state.json', 'v28_rows.json', 'v29_rows.json',
                   'csv_authoring_receipt.json', 'preview.json', 'inspection.json'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})')
SOURCE_FILES = ('src/round111_112_research.py', 'src/round111_residual_prior_train.py',
                'src/round112_ordered_unit.py', 'src/package_research_continue_20261008.py',
                'src/package_research_continue_20261008_v2.py',
                'src/round111_checkpoint_review.py', 'src/round111_112_full_release.py',
                'src/round113_offset_composition.py', 'src/round113_offset_research.py')
REPORT_FILES = ('reports/round111_112_plan.md', 'reports/round111_112_report.md',
                'reports/round113_plan.md', 'reports/round113_report.md')
BATCH_CORE = ('request.json', 'request.sha256', 'seal.json', 'seal.sha256',
              'preflight_review.json', 'preflight_review.sha256', 'postfit_review.json', 'postfit_review.sha256',
              'training_execution.json', 'training_execution.sha256',
              'evaluation_execution.json', 'evaluation_execution.sha256',
              'R111_residual_selfcheck_execution.json', 'R111_residual_selfcheck_execution.sha256',
              'R112_helper_selfcheck_execution.json', 'R112_helper_selfcheck_execution.sha256')
BATCH_OPTIONAL = ('launch.py', 'R111_residual_selfcheck.log', 'R112_helper_selfcheck.log',
                  'freeze.log', 'freeze_execution.json', 'freeze_execution.sha256',
                  'six_fits.log', 'six_fits_execution.json', 'six_fits_execution.sha256',
                  'evaluation.log', 'evaluation_execution.log', 'postfit_review.log',
                  'verify_completed.py', 'verify_completed_v2.py', 'review_final.py',
                  'audit_model_identity.py', 'postfit_schema_selfcheck.py',
                  'postfit_tool_preparation.json', 'postfit_tool_preparation.sha256')
CONTINUE_REQUIRED = ('user_request.json',)
CONTINUE_OPTIONAL = (
    'source_preparation.json', 'prepare_batch_source.py',
    'direction_review_a/direction_review.md', 'direction_review_a/direction_review.json',
    'direction_review_a/direction_review.sha256', 'direction_review_a/round112_helper_selfcheck.json',
    'direction_review_a/direction_build_attempt1_failure.json',
    'direction_review_a/package_author_selfcheck.json',
    'direction_review_a/package_author_review.json',
    'direction_review_a/package_v2_author_selfcheck.json',
    'direction_review_a/package_v2_author_review.json',
    'direction_review_b/R111_method_review.md',
    'direction_review_b/R111_author_selfcheck.json', 'direction_review_b/R111_author_selfcheck.sha256',
    'direction_review_b/R111_author_selfcheck_v2.json', 'direction_review_b/R111_author_selfcheck_v2.sha256',
    'direction_review_b/R111_checkpoint_tool_author_selfcheck.json',
    'direction_review_b/R111_checkpoint_tool_author_selfcheck.sha256',
    'direction_review_b/R111_R112_full_release_author_selfcheck.json',
    'direction_review_b/R111_R112_full_release_author_selfcheck.sha256',
    'pipeline_review/pipeline_review.json', 'pipeline_review/pipeline_review.sha256',
    'pipeline_review/review_workflow.md', 'pipeline_review/audit_primitives.py',
    'pipeline_review/finish_preflight.py', 'pipeline_review/final_preflight_checks.json',
    'pipeline_review/final_preflight_checks.sha256', 'pipeline_review/arithmetic_selfcheck.json',
    'pipeline_review/arithmetic_selfcheck.sha256', 'pipeline_review/preflight_driver_selfcheck.py',
    'pipeline_review/preflight_driver_selfcheck.json', 'pipeline_review/preflight_driver_selfcheck.sha256',
    'pipeline_review/independent_R111_cpu_checks.json',
    'pipeline_review/independent_R111_cpu_checks_v2.json', 'pipeline_review/independent_R112_cpu_checks.json',
    'package_author_review.json', 'package_independent_review.json')
IMPLEMENTATION_PARENT = {'path': 'src/package_research_continue_20261008.py',
                         'sha256': '43fd88200f5cc7d63250ab82347828d499077a7ab5cef8de46559aa9d7707ea1'}
PRODUCTION = (
    'src/house_features.py', 'src/nn_predict.py', 'src/nn_prepare.py', 'src/predict_v22.py',
    'src/round47_nn_encode.py', 'src/round54_numeric_encode.py', 'src/submission_io.py', 'src/tree_runtime.py',
    'src/closure_modules/model_common.py', 'src/closure_modules/prepare_data.py',
    'src/closure_modules/round2_neighborhood.py', 'src/closure_modules/round6_geo.py',
    'src/closure_modules/round8_spatial_reference.py',
)
METHODS = {
    111: 'prior NN固定residual MLP：保持日期總價MAE／prior，檢查內部skip架構',
    112: 'R79單價/date×area：僅Plain→Ordered，檢查target/weight交互；保留R17失敗反證',
}


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


def policy(name, payload=None):
    path = relative(name)
    need(path.suffix.lower() in ALLOWED_SUFFIXES and path.name.lower() not in FORBIDDEN_NAMES,
         'Excluded artifact type/state: ' + name)
    need(path.parts[0] in {'src', 'reports', 'outputs'}, 'Only named source/report/output evidence is allowed')
    if path.parts[0] == 'src':
        need(len(path.parts) == 2 and path.suffix == '.py', 'Only standalone Python sources under src/')
    elif path.parts[0] == 'reports':
        need(len(path.parts) == 2 and re.match(r'round(?:111|112|113)(?:_|\.)', path.name),
             'Only R111--R113 reports are in scope')
    else:
        need(len(path.parts) >= 3, 'Output must name an explicit file below a bounded root')
        allowed_root = (name == EXACT_BUILDER or '/'.join(path.parts[:2]) in GROUPS
                        or bool(re.fullmatch(r'round(?:111|112|113)', path.parts[1])))
        need(allowed_root and len(path.parts) >= 3, 'Output outside the bounded round/request roots')
    if payload is not None:
        text = check_text(payload, name)
        if path.suffix == '.json':
            check_aggregate(json_value(payload))
        elif path.suffix == '.log':
            for line in text.splitlines():
                need(not re.fullmatch(r'\s*\d+\s*,\s*\d+(?:\.\d+)?\s*',line),
                     'CSV-like per-row log data excluded: '+name)
                need(len(line) <= 65536, 'Unexpected long log line; inspect for row data: ' + name)
                if line.lstrip().startswith(('{', '[')):
                    try:
                        value = json_value(line.encode('utf-8'))
                    except (json.JSONDecodeError, UnicodeError):
                        continue
                    check_aggregate(value)
    return path


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


def parse_dispositions(releases, no_releases, deferred_releases=()):
    selected = {}
    explicit = set()
    for kind, values in (('release', releases), ('no_release_reason', no_releases),
                         ('deferred_release_reason', deferred_releases)):
        for specification in values:
            left, separator, right = specification.partition('=')
            need(separator and left.isdigit() and int(left) in ROUNDS and right.strip(),
                 'Expected ROUND=PATH or ROUND=REASON for rounds 111--112')
            number = int(left)
            need(number not in explicit, 'A round disposition was supplied twice')
            explicit.add(number)
            if kind == 'release':
                path = policy(right)
                need(path.suffix == '.json' and path.parts[1] == 'round' + str(number)
                     and path.name == 'release.json', 'Release must name that round release.json')
            else:
                need(len(right.strip()) >= 8, 'Supply a concrete non-release/defer reason')
                check_text(right.encode('utf-8'), 'round disposition reason')
            selected[number] = {kind: right.strip()}
    need(set(selected) == set(ROUNDS), 'Explicit --release, --no-release or --defer-release required for rounds 111,112')
    return selected


def parse_release_reviews(specifications):
    selected = {}
    explicit = set()
    for specification in specifications:
        left, separator, right = specification.partition('=')
        need(separator and left.isdigit() and int(left) in ROUNDS and right.strip(),
             'Expected release review ROUND=PATH for rounds 111--112')
        number = int(left)
        need(number not in explicit, 'A release review was supplied twice')
        explicit.add(number)
        path = policy(right.strip())
        need(path.parts[0] == 'outputs' and path.suffix == '.json', 'Release review must be bounded output JSON')
        selected[number] = right.strip()
    return selected


def check_disposition(number, disposition, results, extra_passed=True):
    passed = results['gates']['exploratory_submission']['passed']
    need(type(passed) is bool and type(extra_passed) is bool, 'Gates must be explicit booleans')
    eligible = passed and extra_passed
    if 'release' in disposition:
        need(eligible, 'Release needs original exploration AND independent R113 stability gates')
        return 'released_exploratory_candidate_not_selected'
    reason_key = 'deferred_release_reason' if eligible else 'no_release_reason'
    need(set(disposition) == {reason_key},
         'Eligible requires defer; failed original/extra gate requires no-release: R'+str(number))
    reason = disposition[reason_key]
    need(len(reason) >= 8, 'Concrete no-release/defer reason required')
    check_text(reason.encode('utf-8'),'round disposition reason')
    need(not safe(ROOT,f'outputs/round{number}/final/release.json').exists(),
         'Completed final release exists; explicitly include independent review')
    need(not list((ROOT/'outputs').glob(f'submission_v{VERSIONS[number]}*.csv')),
         'No-release/defer conflicts with an existing candidate CSV')
    return ('eligible_full_release_deferred' if eligible else
            'extra_gate_failed_no_release' if passed else 'original_exploration_failed_no_release')


def check_model_metadata(number, folder, metadata):
    need(metadata.get('status') == 'completed' and metadata.get('saved_model_reload_exact') is True
         and metadata.get('query_labels_used') is False
         and metadata.get('reload_rows') == metadata.get('query_rows'),
         'Native training/reload/scope incomplete')
    if number == 111:
        cfg, old = metadata['configuration'], metadata['baseline_configuration']
        overrides = metadata['architecture_configuration_overrides']
        need(metadata['complete_R63_nonarchitecture_training_configuration_unchanged'] is True
             and metadata['actual_model_class'] == 'round111_residual_prior_train.TabularResidualPrior'
             and metadata['numeric_count'] == 144 and len(metadata['cardinalities']) == 23
             and cfg == {**old, **overrides}
             and cfg['residual_width'] == 256 and cfg['residual_expansion'] == 512
             and cfg['residual_block_count'] == 2 and cfg['hidden'] == [256,256]
             and cfg['seed'] == 42 and cfg['epochs'] == 30 and cfg['batch_size'] == 2048
             and cfg['loss_formula'] == 'mean(train_sample_weight * abs(ratio_output * train_price_factor_million - train_y_million))'
             and metadata['saved_native_ratio_reload_exact'] is True
             and [v['epoch'] for v in metadata['epochs']] == list(range(1,31)),
             'R111 fixed residual architecture/original total MAE differs')
    elif number == 112:
        config, prior = metadata['parameters'], metadata['baseline_parameters']
        actual = metadata['native_parameters']
        need(set(config) == set(prior)
             and sorted(k for k in config if config[k] != prior[k]) == ['boosting_type']
             and prior['boosting_type'] == 'Plain' and config['boosting_type'] == 'Ordered'
             and 'ctr_target_border_count' not in config
             and metadata['changed_parameter_names'] == ['boosting_type']
             and metadata['added_parameter_names'] == []
             and metadata['actual_boosting_type'] == actual['boosting_type'] == 'Ordered'
             and metadata['actual_task_type'] == actual['task_type'] == 'GPU'
             and metadata['actual_ctr_target_border_count'] == actual['ctr_target_border_count'] == 1
             and metadata['actual_max_ctr_complexity'] == 2 and metadata['actual_model_depth'] == 8
             and metadata['actual_trees'] == 2999 and metadata['feature_count'] == 95
             and metadata['categorical_feature_count'] == 23
             and metadata['raw_and_TWD_reload_exact'] is True
             and metadata['fit_uses_eval_set'] is False and metadata['early_stopping_used'] is False,
             'R112 fitted native recipe differs')
    else:
        need(False, 'Out-of-scope round')


def check_final_source_binding(number, request, review):
    """Complete final closure, separate final plan and independent preflight."""
    expected_plan = f'reports/round{number}_final_plan.md'
    expected_preflight = GROUP + f'/round{number}_final_release_preflight.json'
    need(request['status'] == 'frozen_before_one_exploratory_final_fit'
         and request['round'] == number and request['version'] == VERSIONS[number]
         and request['new_fits'] == 1 and request['query_labels_available'] is False
         and request['selected_submission_preserved'] == 'v22'
         and request['Public_score'] is None and request['Private_score'] is None,
         'Final request fit/label/selection contract differs')
    sources = request['sources']
    need(isinstance(sources, dict) and sources
         and 'src/round111_112_full_release.py' in sources and EXACT_BUILDER in sources
         and review['source_sha256'] == sources,
         'Final review must bind the exact complete frozen source closure, including builder')
    for source, expected in sources.items():
        policy(source)
        need(reference(source)['sha256'] == expected, 'Final frozen source changed: ' + source)
    need(request['paths']['plan'] == expected_plan and request['paths']['preflight'] == expected_preflight,
         'Final plan/preflight path differs')
    for name in (expected_plan, expected_preflight):
        need(request['inputs'][name] == reference(name)['sha256'], 'Final plan/preflight frozen hash differs')
    preflight = read(expected_preflight)
    need(preflight['status'] == 'passed' and preflight['round'] == number
         and preflight['version'] == VERSIONS[number]
         and preflight['source_sha256'] == {**sources, expected_plan: reference(expected_plan)['sha256']},
         'Independent final preflight complete source/plan binding differs')
    return set(sources) | {expected_plan, expected_preflight,
                          str(PurePosixPath(expected_preflight).with_suffix('.sha256'))}


def check_release_review(number, release_name, released, folder, reviews):
    need(number in reviews, 'Missing explicit independent final release review: R' + str(number))
    name = reviews[number]
    review = read(name)
    need(review['status'] == 'passed', 'Independent final release review has not passed: ' + name)
    extra = required_extra_origin(number)
    request = read(released['request']['path'])
    need(released['required_extra_origin'] == extra
         and review['required_extra_origin'] == extra
         and request['required_extra_origin'] == extra,
         'Release/request/final review do not bind exact passed R113 evidence')
    need(review['release'] == {'path': release_name, 'sha256': sha(safe(ROOT, release_name))},
         'Final review is not bound to this exact release')
    for field in ('request', 'seal', 'authoring', 'CSV'):
        need(review[field] == released[field], 'Final review/release identity differs: ' + field)
    metadata_name = folder + '/model/' + ('training.json' if number == 111 else 'metadata.json')
    need(review['training'] == {'path': metadata_name, 'sha256': sha(safe(ROOT, metadata_name))},
         'Final review/training identity differs')
    for field in ('Public_score', 'Private_score'):
        need(field in review and review[field] is None, 'Unexpected or unspecified score in final review')
    need(review['round'] == number and review['version'] == VERSIONS[number]
         and review['selected_submission_preserved'] == 'v22'
         and review['automatic_Kaggle_upload_or_selection'] is False
         and review['query_or_test_labels_loaded'] is False and review['review_new_fits'] == 0
         and review['trainer_saved_model_reload_exact'] is True
         and review['CSV_rows'] == 47073
         and review['CSV_once_rounded_from_unrounded_median'] is True
         and review['all_sample_IDs_prices_order_headers_and_encoding_exact'] is True
         and review['positive_safe_integer_prices'] is True,
         'Final review does not establish exact CSV identity, preserved selection and unknown scores')
    source_names = check_final_source_binding(number, request, review)
    expected_csv = f"outputs/submission_v{released['version']}_trial.csv"
    need(released['CSV']['path'] == expected_csv and sha(safe(ROOT, expected_csv)) == released['CSV']['sha256'],
         'Actual CSV bytes differ from independent review/release; CSV is hashed, never copied/decoded')
    return name, source_names


def authoring_aggregate(authoring, source_name, source_sha256, expected_csv_sha256):
    """Pure allowlist projection; never return inspection/preview row payloads."""
    keys = ('status', 'rows', 'columns', 'input_json_sha256', 'csv_sha256',
            'full_workbook_values_exact', 'formula_count', 'utf8_bom', 'csv_encoding')
    need(all(key in authoring for key in keys), 'CSV authoring receipt lacks required aggregate fields')
    projection = {key: authoring[key] for key in keys}
    need(type(projection['rows']) is int and projection['rows'] == 47073
         and projection['columns'] == ['編號', '總價元'] and projection['full_workbook_values_exact'] is True
         and projection['csv_sha256'] == expected_csv_sha256 and projection['formula_count'] == 0
         and projection['utf8_bom'] is True, 'CSV authoring aggregate identity mismatch')
    projection.update(source_receipt={'path': source_name, 'sha256': source_sha256},
                      original_receipt_copied=False, inspection_and_preview_rows_included=False,
                      transformation='Allowlisted aggregate receipt; embedded inspection/preview omitted')
    check_aggregate(projection)
    policy(str(PurePosixPath(source_name).with_name('csv_authoring_aggregate.json')), encoded(projection))
    return projection


def check_extra_postfit(review, req, seal, results, gate):
    """Independent R113 receipt must establish this exact completed batch."""
    required_records = {'request': 'outputs/round113/request.json', 'seal': 'outputs/round113/seal.json',
        'training_execution': EXTRA_GROUP + '/training_execution.json',
        'evaluation_execution': EXTRA_GROUP + '/evaluation_execution.json',
        'results': EXTRA_RESULTS, 'release_gate': EXTRA_GATE, 'plan': EXTRA_PLAN}
    need(review['status'] == 'passed' and review['protected_unchanged'] is True
         and review['new_model_fits'] == 0 and review['GPU_work'] is False
         and review['Private_score'] is None and review['Public_score'] is None
         and review['blocking_findings_remaining'] == [] and review['selected_submission'] == 'v22'
         and review['new_CSV'] is False, 'R113 independent postfit completion/privacy/protection flags differ')
    names = set()
    for field, path in required_records.items():
        need(review[field] == reference(path), 'R113 independent postfit exact record differs: ' + field)
    check_record(review['review_tool']); names.add(review['review_tool']['path'])
    candidates = {'107', '111', '112'}
    need(set(review['artifacts']) == set(review['trainmeta']) == set(review['manual_gate_verdicts']) == candidates,
         'R113 independent review must cover exactly all three candidates')
    for name, expected in req['protected_before'].items():
        need(reference(name)['sha256'] == expected, 'R113 actual protected artifact changed: ' + name)
    for candidate in sorted(candidates):
        folder = 'outputs/round113/R' + candidate
        stage = read(folder + '/stage.json'); actual = results['candidates'][candidate]
        audited = review['artifacts'][candidate]
        need(set(audited) == {'stage', 'training_metadata', 'evaluation', 'sealed_models'}
             and audited['stage'] == seal['records'][candidate] == reference(folder + '/stage.json')
             and audited['training_metadata'] == stage['model_metadata']
             and audited['evaluation'] == reference(folder + '/evaluation/results.json')
             and audited['sealed_models'] == stage['sealed_models']
             and review['trainmeta'][candidate] == read(stage['model_metadata']['path']),
             'R113 independent stage/all-model/metadata/evaluation binding differs: ' + candidate)
        need(stage['sealed_models'], 'R113 native model set cannot be empty')
        manual = review['manual_gate_verdicts'][candidate]
        need(set(manual) == {'components', 'passed', 'both_point_metrics_improve'}
             and all(type(v) is bool for v in manual['components'].values())
             and all(type(v) is bool for v in actual['gate']['components'].values())
             and manual['components'] == actual['gate']['components'] == gate['candidate_gates'][candidate]['components']
             and manual['passed'] is actual['gate']['passed'] is gate['candidate_gates'][candidate]['passed']
             and type(manual['both_point_metrics_improve']) is bool
             and manual['both_point_metrics_improve'] is actual['both_point_metrics_improve'],
             'R113 independently recomputed manual gate differs: ' + candidate)
    return names


def extra_evidence():
    """R113 is an additional origin, never a fourth tuned model or CSV round."""
    names={EXTRA_GATE,EXTRA_REVIEW,EXTRA_PLAN,EXTRA_RESULTS,
           EXTRA_GROUP+'/round113_release_gate.sha256', EXTRA_GROUP+'/preflight_review.json',
           EXTRA_GROUP+'/preflight_review.sha256',EXTRA_GROUP+'/postfit_review.sha256'}
    names.update('outputs/round113/'+x for x in ('request.json','request.sha256','seal.json','seal.sha256','results.sha256'))
    gate=read(EXTRA_GATE);req=read('outputs/round113/request.json')
    need(reference(EXTRA_PLAN)['sha256']==EXTRA_PLAN_SHA256, 'Fixed pre-scoring R113 plan identity differs')
    need(gate['status']=='fixed_extra_origin_evaluated' and gate['candidates']==[107,111,112]
         and set(gate['candidate_gates'])=={'107','111','112'}
         and gate['Private_score'] is None and gate['Public_score'] is None
         and gate['selected_submission']=='v22' and gate['new_CSV'] is False,
         'R113 aggregate gate incomplete or scope differs')
    need(gate['plan']==reference(EXTRA_PLAN) and gate['results']==reference(EXTRA_RESULTS),
         'R113 gate plan/results identity differs')
    keys={'overall_regression_at_most_0_25pct','weighted_regression_at_most_0_25pct',
          'no_quarter_or_large_group_above_5pct'}
    for candidate,item in gate['candidate_gates'].items():
        need(set(item['components'])==keys and all(type(v) is bool for v in item['components'].values())
             and type(item['passed']) is bool and item['passed']==all(item['components'].values()),
             'R113 stability gate is not explicit three-component rule')
    need(isinstance(req['source_sha256'],dict) and req['source_sha256'], 'R113 source closure missing')
    need(req['status']=='frozen_before_three_offset_fits' and req['candidates']==[107,111,112]
         and req['new_fits']==3 and req['query_scores_computed'] is False
         and req['query_labels_used_for_fit'] is False and req['plan']==reference(EXTRA_PLAN)
         and req['fit_rows']==272117 and req['query_rows']==107169
         and req['selected_submission']=='v22' and req['new_CSV'] is False
         and req['Public_score'] is None and req['Private_score'] is None,
         'R113 request fixed scope differs')
    seal=read('outputs/round113/seal.json');results=read(EXTRA_RESULTS)
    need(seal['status']=='three_offset_models_all_query_predictions_sealed_unscored'
         and seal['request']==reference('outputs/round113/request.json')
         and seal['query_scores_computed'] is False and set(seal['records'])=={'107','111','112'},
         'R113 pre-scoring seal differs')
    need(results['status']=='fixed_extra_origin_evaluated'
         and results['request']==reference('outputs/round113/request.json')
         and results['seal']==reference('outputs/round113/seal.json')
         and results['plan']==reference(EXTRA_PLAN) and results['source_sha256']==req['source_sha256']
         and set(results['candidates'])=={'107','111','112'}
         and results['query_labels_used_for_fit'] is False
         and results['Public_score'] is None and results['Private_score'] is None
         and results['selected_submission']=='v22' and results['new_CSV'] is False,
         'R113 result/request/seal/source scope differs')
    for candidate in (107,111,112):
        directory=f'outputs/round113/R{candidate}'
        stage=read(directory+'/stage.json');record=reference(directory+'/stage.json')
        need(seal['records'][str(candidate)]==record
             and results['candidates'][str(candidate)]['sealed_stage']==record
             and stage['candidate_round']==candidate and stage['stage']=='offset_2023_02_11'
             and stage['fit_rows']==272117 and stage['query_rows']==107169
             and stage['query_labels_available'] is False
             and results['candidates'][str(candidate)]['gate']==gate['candidate_gates'][str(candidate)],
             'R113 candidate stage/seal/gate identity differs')
        metadata_name=f'outputs/round113/models/R{candidate}/'+('metadata.json' if candidate==112 else 'training.json')
        need(stage['model_metadata']==reference(metadata_name), 'R113 native metadata identity differs')
        for model in stage['sealed_models']:check_record(model)
        names.update(directory+'/'+x for x in ('stage.json','stage.sha256','train.log',
                                                'evaluation/results.json','evaluation/results.sha256'))
        names.add(metadata_name)
        names.add(f'outputs/round113/models/R{candidate}/'+('started.json' if candidate==112 else 'run.json'))
        need(read(directory+'/evaluation/results.json')==results['candidates'][str(candidate)],
             'R113 candidate and aggregate evaluations differ')
    for short in ('training_execution','evaluation_execution'):
        name=EXTRA_GROUP+'/'+short+'.json'
        need(read(name)['exit_code']==0, 'R113 execution incomplete')
        names.update((name,str(PurePosixPath(name).with_suffix('.sha256'))))
    for prefix in ('R107_loss','R111_residual','R112_helper','R113_driver'):
        for suffix in ('_selfcheck.log','_selfcheck_execution.json','_selfcheck_execution.sha256'):
            names.add(EXTRA_GROUP+'/'+prefix+suffix)
    for name in ('launch.py','freeze.log','freeze_execution.json','freeze_execution.sha256',
                 'three_fits.log','three_fits_execution.json','three_fits_execution.sha256',
                 'evaluation.log','postfit_review.log','verify_completed.py',
                 'adapter_author_preflight.json','adapter_author_preflight.sha256',
                 'adapter_author_preflight.log','adapter_author_preflight_execution.json'):
        if safe(ROOT,EXTRA_GROUP+'/'+name).is_file():names.add(EXTRA_GROUP+'/'+name)
    expected={**req['source_sha256'],EXTRA_PLAN:reference(EXTRA_PLAN)['sha256']}
    for short in ('preflight_review','postfit_review'):
        review=read(EXTRA_GROUP+'/'+short+'.json')
        need(review['status']=='passed' and review['source_sha256']==expected,
             'R113 independent source/plan review closure differs')
        for source,value in expected.items():
            need(reference(source)['sha256']==value,'R113 frozen source/plan changed')
            names.add(source)
    review=read(EXTRA_REVIEW)
    need(review['release_gate']==reference(EXTRA_GATE) and review['results']==reference(EXTRA_RESULTS)
         and review['plan']==reference(EXTRA_PLAN), 'R113 postfit is not bound to actual gate/results/plan')
    names.update(check_extra_postfit(review, req, seal, results, gate))
    return gate,names


def required_extra_origin(number):
    gate,_=extra_evidence()
    candidate=gate['candidate_gates'][str(number)]
    need(candidate['passed'] is True,'R113 extra origin failed; no new full release')
    return {'gate':reference(EXTRA_GATE),'review':reference(EXTRA_REVIEW),
            'results':reference(EXTRA_RESULTS),'plan':reference(EXTRA_PLAN),'candidate_gate':candidate}


def check_sidecar(name,payload):
    multi=CONTINUE_GROUP+'/direction_review_a/direction_review.sha256'
    if name==multi:
        found=set()
        for line in payload.decode('ascii').splitlines():
            match=re.fullmatch(r'([0-9a-f]{64})  ([A-Za-z0-9_.-]+)',line)
            need(match is not None,'Invalid direction multi-file SHA record')
            digest,filename=match.groups();need(filename not in found,'Duplicate sidecar member')
            found.add(filename)
            need(sha(safe(ROOT,str(PurePosixPath(name).parent/filename)))==digest,'Multi-file SHA differs')
        need(found=={'direction_review.json','direction_review.md','round112_helper_selfcheck.json'},
             'Direction multi-file sidecar membership differs')
    else:
        counterpart=safe(ROOT,name).with_suffix('.json')
        # Either the original bare digest or a single exact same-directory JSON
        # filename. Never accept arbitrary sidecar paths/multiple files.
        text=payload.decode('ascii').strip()
        match=re.fullmatch(r'([0-9a-f]{64})(?:[ \t]+([^\r\n]+))?',text)
        need(counterpart.is_file() and match is not None
             and (match.group(2) is None or match.group(2)==counterpart.name),
             'Expected single-SHA JSON sidecar')
        need(match.group(1)==sha(counterpart),'SHA sidecar differs: '+name)


def inventory(dispositions, extras, reviews):
    """Exact named evidence only: no outputs rglob or automatic model folder copy."""
    extra_gate, extra_names = extra_evidence()
    names = set(SOURCE_FILES) | set(REPORT_FILES) | extra_names
    names.update(GROUP + '/' + name for name in BATCH_CORE)
    names.update(CONTINUE_GROUP + '/' + name for name in CONTINUE_REQUIRED)
    for group, options in ((GROUP,BATCH_OPTIONAL),(CONTINUE_GROUP,CONTINUE_OPTIONAL)):
        names.update(group+'/'+name for name in options if safe(ROOT,group+'/'+name).is_file())
    request = read(GROUP+'/request.json')
    need(request['status'] == 'frozen_before_six_fits' and request['query_scores_computed'] is False
         and request['new_fits'] == 6 and request['public_tuning'] is False
         and request['private_score'] is None, 'Request is not the fixed six-fit unscored contract')
    check_record(request['plan'], 'reports/round111_112_plan.md')
    sources = request['source_sha256']
    need(isinstance(sources,dict) and sources, 'Missing frozen source closure')
    for name, expected in sources.items():
        need(name.startswith('src/') and reference(name)['sha256'] == expected, 'Frozen source changed')
        names.add(name)
    seal = read(GROUP+'/seal.json')
    need(seal['status'] == 'six_models_all_query_predictions_sealed_unscored'
         and seal['request'] == reference(GROUP+'/request.json')
         and seal['query_scores_computed'] is False, 'Incomplete pre-score seal')
    pre = read(GROUP+'/preflight_review.json')
    post = read(GROUP+'/postfit_review.json')
    need(pre['status'] == post['status'] == 'passed', 'Independent preflight/postfit must pass')
    need(pre['source_sha256'] == {**sources,request['plan']['path']:request['plan']['sha256']},
         'Preflight source/plan closure differs')
    need(post['source_sha256'] == {**sources,request['plan']['path']:request['plan']['sha256']}
         and post['plan'] == request['plan']
         and post['request'] == reference(GROUP+'/request.json') and post['seal'] == reference(GROUP+'/seal.json')
         and post['new_model_fits'] == 0 and post['GPU_work'] is False
         and post['Private_score'] is None and post['blocking_findings_remaining'] == []
         and post['protected_unchanged'] is True, 'Postfit does not bind actual request/seal/source/protected state')
    names.add(post['review_tool']['path']);check_record(post['review_tool'])
    for name, expected in request['protected_before'].items():
        need(reference(name)['sha256'] == expected, 'Actual protected original changed: '+name)
    for short in ('training_execution','evaluation_execution'):
        name = GROUP+'/'+short+'.json'
        need(read(name)['exit_code'] == 0 and post[short] == reference(name), 'Incomplete/unbound execution')
    projections = []
    for number in ROUNDS:
        directory = f'outputs/round{number}'
        names.update((directory+'/results.json',directory+'/results.sha256'))
        results = read(directory+'/results.json')
        need(results['status'] == 'three_reused_windows_evaluated' and set(results['stages']) == set(STAGES)
             and results.get('Private_score') is None, 'Incomplete round evaluation')
        need(post['results'][str(number)] == reference(directory+'/results.json'), 'Postfit/result identity differs')
        for field in ('request','seal'):
            need(results[field] == reference(GROUP+'/'+field+'.json'), 'Results point to another frozen batch')
        manual = post['manual_gate_verdicts'][str(number)]
        gates = results['gates']
        need(manual['components'] == gates['exploratory_submission']['components']
             and manual['exploratory_submission_eligible'] is gates['exploratory_submission']['passed']
             and manual['strong_components'] == gates['promotion']['components']
             and manual['strong_research_gate_passed'] is gates['promotion']['adoption_research_gate_passed']
             and manual['confirmation_fractional_improvement'] == gates['promotion']['confirmation_fractional_improvement'],
             'Independent manual gate differs from evaluated result')
        for stage in STAGES:
            folder = directory+'/'+stage
            names.update(folder+'/'+name for name in ('stage.json','stage.sha256','train.log',
                                                    'evaluation/results.json','evaluation/results.sha256'))
            model_names = ('run.json','training.json') if number == 111 else ('started.json','metadata.json')
            names.update(folder+'/model/'+name for name in model_names)
            metadata_name = folder+'/model/'+('training.json' if number == 111 else 'metadata.json')
            stage_record = read(folder+'/stage.json')
            need(seal['records'][str(number)][stage] == reference(folder+'/stage.json')
                 and stage_record['candidate_round'] == number and stage_record['stage'] == stage
                 and stage_record['query_labels_available'] is False, 'Stage seal/scope differs')
            need(stage_record['model_metadata'] == reference(metadata_name), 'Stage metadata differs')
            evaluation_name = folder+'/evaluation/results.json'
            need(read(evaluation_name) == results['stages'][stage], 'Stage/aggregate evaluations differ')
            training = read(metadata_name);check_model_metadata(number,folder,training)
            audited = post['artifacts'][str(number)+'/'+stage]
            need(audited['stage'] == reference(folder+'/stage.json')
                 and audited['training_metadata'] == reference(metadata_name)
                 and audited['evaluation'] == reference(evaluation_name)
                 and post['trainmeta'][str(number)+'/'+stage] == training, 'Postfit stage/model binding differs')
            for record in stage_record['sealed_models']:
                check_record(record)
            for record in audited['sealed_models']:
                check_record(record)
                need(record in stage_record['sealed_models'], 'Reviewed model absent from original seal')
        check_disposition(number,dispositions[number],results,extra_gate['candidate_gates'][str(number)]['passed'])
        if 'release' in dispositions[number]:
            release_name = dispositions[number]['release'];released = read(release_name)
            need(released['status'] == 'exploratory_submission_ready_not_selected'
                 and released['round'] == number and released['version'] == VERSIONS[number]
                 and released['rows'] == 47073 and released['Public_score'] is None and released['Private_score'] is None
                 and released['selected_submission_preserved'] == 'v22'
                 and released['automatic_Kaggle_upload_or_selection'] is False, 'Final release identity/scope differs')
            folder = str(PurePosixPath(release_name).parent)
            for name in ('request.json','request.sha256','seal.json','seal.sha256','release.json','release.sha256',
                         'fit_execution.json','fit_execution.sha256','train.log','csv_authoring.log'):
                names.add(folder+'/'+name)
            model_names = ('run.json','training.json') if number == 111 else ('started.json','metadata.json')
            names.update(folder+'/model/'+name for name in model_names)
            training_name=folder+'/model/'+('training.json' if number==111 else 'metadata.json')
            metadata=read(training_name);check_model_metadata(number,folder,metadata)
            need(metadata['fit_rows']==467221 and metadata['query_rows']==47073
                 and read(folder+'/fit_execution.json')['exit_code']==0, 'Fullfit incomplete')
            for field in ('request','seal'):
                need(released[field]==reference(folder+'/'+field+'.json'), 'Final request/seal differs')
            authoring_name=folder+'/csv_authoring_receipt.json'
            payload=safe(ROOT,authoring_name).read_bytes();check_text(payload,authoring_name)
            need(released['authoring']=={'path':authoring_name,'sha256':sha_bytes(payload)}, 'Authoring identity differs')
            projection=authoring_aggregate(json_value(payload),authoring_name,sha_bytes(payload),released['CSV']['sha256'])
            projections.append({'source':authoring_name,'source_sha256':sha_bytes(payload),
                                'destination':'evidence/'+folder+'/csv_authoring_aggregate.json','payload':encoded(projection)})
            name,final_source_names=check_release_review(number,release_name,released,folder,reviews)
            names.update(final_source_names)
            names.update((name,str(PurePosixPath(name).with_suffix('.sha256'))))
    names.update(extras)
    missing=[name for name in sorted(names) if not safe(ROOT,name).is_file()]
    need(not missing,'Missing required completion evidence; parent must resolve actual status:\n'+'\n'.join(missing))
    entries=[]
    for name in sorted(names):
        payload=safe(ROOT,name).read_bytes();policy(name,payload)
        if name.endswith('.sha256'):
            check_sidecar(name,payload)
        entries.append({'source':name,'destination':destination(name),'bytes':len(payload),'sha256':sha_bytes(payload)})
    return entries,projections


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
    need(sha_bytes(old_bytes) == BASE_MANIFEST_SHA256, 'Original 3017-member manifest identity changed; review before packaging')
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


def summary(dispositions):
    lines = ['# 2026-10-08 恢復研究：R111–R112', '',
             '使用者重新授權改良模型及新試交。v22與原簡報保留；v22已回報Public MAE為1,998,568.5706元，Private未知。',
             '本輪停止線為Codex剩餘20%，不購買點數或自行重置；新研究不取代v22選定。',
             '以下為反覆使用歷史三窗的離線MAE；新候選Public／Private均未知，沒有代替使用者上傳或勾選Kaggle。', '',
             '|方法與固定理由|development ΔMAE|confirmation ΔMAE|auxiliary ΔMAE|探索gate|強gate|產物終態|',
             '|---|---:|---:|---:|---|---|---|']
    extra_gate, _ = extra_evidence()
    for number in ROUNDS:
        result = read(f'outputs/round{number}/results.json')
        deltas = []
        for stage in STAGES:
            metrics = result['stages'][stage]['metrics']
            difference = metrics['T']['overall_mae'] - metrics['v22']['overall_mae']
            need(math.isfinite(difference), 'Nonfinite summary metric')
            deltas.append(f'{difference:+,.2f}')
        gates = result['gates']
        status = dispositions[number]
        if 'release' in status:
            release = read(status['release'])
            role = f"v{release['version']}試交候選，未選定"
        else:
            extra_passed=extra_gate['candidate_gates'][str(number)]['passed']
            disposition_status=check_disposition(number,status,result,extra_passed)
            reason_key='deferred_release_reason' if disposition_status=='eligible_full_release_deferred' else 'no_release_reason'
            prefixes={'eligible_full_release_deferred':'原三窗及額外origin通過，尚未完成全量發行：',
                      'extra_gate_failed_no_release':'原三窗探索通過，R113額外origin失敗未發行：',
                      'original_exploration_failed_no_release':'原三窗探索未通過，未發行：'}
            role=prefixes[disposition_status]+status[reason_key].replace('|','／').replace('\n',' ')
        link = f'../evidence/outputs/round{number}/results.json'
        lines.append(f"|[R{number}]({link})：{METHODS[number]}|" + '|'.join(deltas)
                     + f"|{gates['exploratory_submission']['passed']}|{gates['promotion']['adoption_research_gate_passed']}|{role}|")
    lines.extend(['', 'ΔMAE單位為新臺幣元，負值表示優於保存v22。完整整體／新舊加權、分群、五層、v24反證與配對區間在原results及報告；本頁不省略失敗而改稱所有方法成功。', '',
        '探索gate允許每窗指標至多0.25%退步，要求最新兩指標不退步、至少兩窗整體改善且無季度／大型群超5%退步；強gate要求三窗整體及加權均改善、最新整體至少0.5%，同樣限制分群退步。通過探索不等於強推薦或Private較佳。', '',
        '流程固定先提出假設與切點，再核canonical索引／九鍵NN輸入／fit-only權重與OOF價格先驗、各切點狀態，序列fit、封存模型與query向量，之後才讀query真價評分。各研究獨立，未把不同輪變因合併，也未依Public搜尋參數。', '',
        '新提交格式：47,073列，欄名`編號,總價元`；依原sample ID順序，總價於最後一次取整並至少1元。程式／證據包僅保留提交SHA及格式收據，沒有CSV、逐筆價格、NPZ／NPY、模型、encoder／prior／reference狀態或課程原表。', '',
        '限制：重用驗證窗口仍有累積選型偏差；條件bootstrap不是Private勝率；GPU固定seed不保證重訓逐bytes相同。來源可追溯及保存重載驗證，不等於新電腦從零端到端重訓已完成。未到內政部成交系統對答案。', '',
        'R113是固定R97非8月起點的額外穩定性檢查（R107/R111/R112各一次；2NN+1CB），重疊歷史仍非盲測。新全量候選需同時通過三窗與R113的獨立審查；原三窗過而R113失敗明列extra_gate_failed_no_release，不冒稱原gate失敗。既有v25只新增後續證據、不回寫原release。', '',
        f'[本輪精確複製清單](../{COPY_MANIFEST})；原379歷史manifest及所有舊收據保持。README／research README／HANDOFF的舊「停止研究」說法屬其原日期，請以本頁及本輪request／release為新增現況。', ''])
    payload = '\n'.join(lines).encode('utf-8')
    check_text(payload, SUMMARY)
    return payload


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
    replacement['research_continue_20261008'] = {'copy_manifest': COPY_MANIFEST, 'publication_not_performed': True}
    target = REPO / 'SOURCE_MANIFEST.json'
    temporary = target.with_name('SOURCE_MANIFEST.continue_20261008.tmp')
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


def run(dispositions, extras, apply, reviews):
    need(REPO.is_dir(), 'Existing v22 GitHub package is required')
    check_git_base()
    current = package_files()
    template, protected = protect_existing(current)
    entries, projections = inventory(dispositions, extras, reviews)
    need(not (REPO / COPY_MANIFEST).exists() and not (REPO / SUMMARY).exists(),
         'Preserve prior packaging; do not overwrite a completed snapshot')
    document = summary(dispositions)
    planned_paths = [item['destination'] for item in entries] + [item['destination'] for item in projections]
    planned_paths += [SUMMARY,COPY_MANIFEST] + [SNAPSHOT+'/before/'+name for name in ('SOURCE_MANIFEST.json',)+NAVIGATION]
    for name in planned_paths:
        safe(REPO,name)
    for item in entries:
        need(not (REPO / item['destination']).exists() or item['destination'] in current,
             'Unlisted destination already exists; main thread must resolve its status: ' + item['destination'])
        if item['destination'] in current:
            need(current[item['destination']]['sha256'] == item['sha256'], 'Existing research/evidence copy differs: ' + item['destination'])
    need(all(not (REPO / item['destination']).exists() for item in projections), 'Preserve any prior aggregate projection')
    planned = {'status': 'preflight_passed_no_writes' if not apply else 'ready_to_copy',
               'files': len(entries), 'bytes': sum(item['bytes'] for item in entries),
               'aggregate_receipt_projections': len(projections),
               'round_dispositions': dispositions, 'release_reviews': reviews,
               'required_extra_origin_gate': reference(EXTRA_GATE),
               'computed_round_status': {n:check_disposition(n,dispositions[n],read(f'outputs/round{n}/results.json'),
                   read(EXTRA_GATE)['candidate_gates'][str(n)]['passed']) for n in ROUNDS},
               'model_fits': 0, 'git_or_network': False}
    if not apply:
        return planned
    for item in entries:
        source = safe(ROOT, item['source'])
        payload = source.read_bytes()
        need(sha_bytes(payload) == item['sha256'], 'Source changed after preflight: ' + item['source'])
        target = safe(REPO, item['destination'])
        if not target.exists():
            write_new(target, payload)
        need(target.read_bytes() == payload, 'Exact copied bytes differ')
    write_new(REPO / SUMMARY, document)
    generated = [{'path': SUMMARY, 'sha256': sha_bytes(document), 'bytes': len(document)}]
    for item in projections:
        need(sha(safe(ROOT, item['source'])) == item['source_sha256'], 'Original authoring receipt changed after preflight')
        write_new(REPO / item['destination'], item['payload'])
        generated.append({'path': item['destination'], 'sha256': sha_bytes(item['payload']),
                          'bytes': len(item['payload']), 'source': item['source'],
                          'source_sha256': item['source_sha256'],
                          'transformation': 'Allowlisted aggregate; not a byte-identical copy of the original receipt'})
    for name in ('SOURCE_MANIFEST.json',) + NAVIGATION:
        payload = (REPO / name).read_bytes()
        archive = SNAPSHOT + '/before/' + name
        write_new(REPO / archive, payload)
        generated.append({'path': archive, 'sha256': sha_bytes(payload), 'bytes': len(payload)})
    verify_protected(protected)
    receipt = {'status': 'copied_exact_bytes_local_only', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
               'files': entries, 'selected_count': len(entries),
               'copied_count': sum(item['destination'] not in current for item in entries),
               'verified_equal_existing_count': sum(item['destination'] in current for item in entries),
               'copied_bytes': sum(item['bytes'] for item in entries if item['destination'] not in current),
               'generated': generated, 'round_dispositions': dispositions, 'release_reviews': reviews,
               'required_extra_origin_gate': reference(EXTRA_GATE),
               'computed_round_status': planned['computed_round_status'],
               'protected_before': protected,
               'historical_379_manifest_unchanged': True, 'production_13_sources_unchanged': True,
               'all_other_preexisting_package_files_unchanged': True,
               'excluded': ['model/array/CSV/private-state/raw-course files', 'node_modules',
                            'embedded CSV rows or long numeric data vectors', 'credentials'],
               'navigation_finish_separate': True, 'new_model_fits': 0, 'git_or_network': False,
               'clean_environment_end_to_end_retraining_claimed': False}
    write_new(REPO / COPY_MANIFEST, encoded(receipt))
    additions = [item['destination'] for item in entries] + [item['path'] for item in generated] + [COPY_MANIFEST]
    refreshed = refresh_top(template, additions)
    verify_protected(protected)
    return {**planned, 'status': receipt['status'], 'copy_manifest': COPY_MANIFEST,
            'copy_manifest_sha256': sha(REPO / COPY_MANIFEST), 'SOURCE_MANIFEST': refreshed,
            'finish': 'Add dated navigation notes preserving original bodies, then --refresh-manifest; publication is separate.'}


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


def self_check():
    """Pure in-memory policy/contract tests; no copy/git/fit/GPU or FS writes."""
    checks=[]
    def reject(name,call):
        try:call()
        except (ValueError,KeyError):checks.append(name)
        else:raise AssertionError('Guard accepted '+name)
    check_aggregate({'rows':47073,'CSV':{'path':'outputs/submission_v28_trial.csv','sha256':'a'*64},
                     'epochs':[{'epoch':n} for n in range(30)],'cardinalities':list(range(23))})
    checks.append('aggregate_counts_references_and_epoch_logs_allowed')
    for name in ('outputs/round111/model.pt','outputs/round112/model.cbm','outputs/round111/a.npy',
                 'outputs/round111/a.npz','outputs/round111/a.pkl','outputs/round111/a.csv','outputs/round111/a.xlsx',
                 'outputs/round111/feature_state.json','outputs/round112/encoder_state.json',
                 'outputs/research_csv_intermediate_20261008/v28_rows.json','outputs/round111/node_modules/x.json',
                 'outputs/round110/results.json','outputs','outputs/round111','', '.',
                 'outputs/research_resume_20261008/arbitrary.json','../outside.json','C:/outside.json'):
        reject('excluded_path:'+name,lambda name=name:policy(name))
    for value in ({'rows':[[1,100]]},{'predictions':[1,2]},{'values':list(range(256))},
                  {'inspection':'private rows'},{'preview':{'id':'x'}},{'state_dict':{'weight':[1,2]}},
                  {'cells':{'project':{'median':2}}},{'any':[{'sample_id':'x','price_twd':123}]}):
        reject('embedded_data_or_state',lambda value=value:check_aggregate(value))
    disposition=parse_dispositions(['111=outputs/round111/final/release.json'],['112=failed fixed gates'])
    need(disposition[111]['release'].endswith('release.json'),'Explicit release parse')
    reject('disposition_missing_round',lambda:parse_dispositions([],['112=failed gates']))
    reject('disposition_duplicate',lambda:parse_dispositions([],['111=failed gates','111=failed gates','112=failed gates']))
    for passed,kind in ((True,'no_release_reason'),(False,'deferred_release_reason')):
        reject('gate_disposition_mismatch',lambda passed=passed,kind=kind:check_disposition(111,
               {kind:'a concrete reason'},{'gates':{'exploratory_submission':{'passed':passed}}}))
    reject('missing_final_independent_review',lambda:check_release_review(111,'outputs/round111/final/release.json',{},'',{}))
    authoring={'status':'artifact_typed_range_exported_to_csv','rows':47073,'columns':['編號','總價元'],
               'input_json_sha256':'a'*64,'csv_sha256':'b'*64,'full_workbook_values_exact':True,
               'formula_count':0,'utf8_bom':True,'csv_encoding':'comma-delimited CRLF',
               'inspection':'{private-ID-and-price}'}
    aggregate=authoring_aggregate(authoring,'outputs/round111/final/csv_authoring_receipt.json','c'*64,'b'*64)
    need('inspection' not in aggregate and b'private-ID' not in encoded(aggregate),'Inspection omitted')
    checks.append('CSV_authoring_allowlist_omits_rows')
    token=('github_'+'pat_'+'A'*30).encode('ascii')
    original=b'Original body\n'
    check_navigation_note(original,b'2026-10-08 dated note\n'+original,'synthetic-nav')
    check_navigation_note(original,original+b'2026-10-08 dated note\n','synthetic-nav')
    checks.append('navigation_prefix_suffix_preserve_body')
    reject('generated_secret',lambda:check_text(token,'synthetic'))
    reject('navigation_secret',lambda:check_navigation_note(original,b'2026-10-08 '+token+original,'synthetic'))
    reject('navigation_body_changed',lambda:check_navigation_note(original,b'2026-10-08 replacement','synthetic'))
    reject('navigation_undated',lambda:check_navigation_note(original,b'undated'+original,'synthetic'))
    reject('duplicate_json_key',lambda:json_value(b'{"status":1,"status":2}'))
    reject('nonfinite_json',lambda:json_value(b'{"metric":NaN}'))
    # Minimal synthetic v2 index proves name and checksum parsing without Git.
    path=b'README.md';entry=b'\0'*60+struct.pack('!H',len(path))+path+b'\0'
    entry+=b'\0'*((-len(entry))%8)
    raw=struct.pack('!4sII',b'DIRC',2,1)+entry;raw+=hashlib.sha1(raw).digest()
    need(git_index_paths(raw)==['README.md'],'Synthetic v2 index parse')
    checks.append('stdlib_Git_index_names_and_checksum')
    reject('index_checksum_mutation',lambda:git_index_paths(raw[:-1]+b'x'))
    from unittest.mock import patch
    def rec(name):return {'path':name,'sha256':EXTRA_PLAN_SHA256 if name==EXTRA_PLAN else sha_bytes(name.encode('utf8'))}
    gate={'status':'fixed_extra_origin_evaluated','candidates':[107,111,112],
          'candidate_gates':{str(n):{'passed':True,'components':{
            'overall_regression_at_most_0_25pct':True,'weighted_regression_at_most_0_25pct':True,
            'no_quarter_or_large_group_above_5pct':True}} for n in (107,111,112)},
          'plan':rec(EXTRA_PLAN),'results':rec(EXTRA_RESULTS),'Private_score':None,'Public_score':None,
          'selected_submission':'v22','new_CSV':False}
    source={'src/round113_offset_research.py':rec('src/round113_offset_research.py')['sha256']}
    closure={**source,EXTRA_PLAN:rec(EXTRA_PLAN)['sha256']}
    pre={'status':'passed','source_sha256':closure}
    post={**pre,'release_gate':rec(EXTRA_GATE),'results':rec(EXTRA_RESULTS),'plan':rec(EXTRA_PLAN)}
    post.update(request=rec('outputs/round113/request.json'),seal=rec('outputs/round113/seal.json'),
        training_execution=rec(EXTRA_GROUP+'/training_execution.json'),
        evaluation_execution=rec(EXTRA_GROUP+'/evaluation_execution.json'),
        review_tool=rec(EXTRA_GROUP+'/verify_completed.py'),artifacts={},trainmeta={},manual_gate_verdicts={},
        protected_unchanged=True,new_model_fits=0,GPU_work=False,Private_score=None,Public_score=None,
        blocking_findings_remaining=[],selected_submission='v22',new_CSV=False)
    data={EXTRA_GATE:gate,'outputs/round113/request.json':{
          'source_sha256':source,'status':'frozen_before_three_offset_fits','candidates':[107,111,112],
          'new_fits':3,'query_scores_computed':False,'query_labels_used_for_fit':False,'plan':rec(EXTRA_PLAN),
          'fit_rows':272117,'query_rows':107169,'selected_submission':'v22','new_CSV':False,
          'Public_score':None,'Private_score':None,'protected_before':{}},
          'outputs/round113/seal.json':{'status':'three_offset_models_all_query_predictions_sealed_unscored',
          'request':rec('outputs/round113/request.json'),'query_scores_computed':False,
          'records':{str(n):rec(f'outputs/round113/R{n}/stage.json') for n in (107,111,112)}},
          EXTRA_RESULTS:{'status':'fixed_extra_origin_evaluated','request':rec('outputs/round113/request.json'),
          'seal':rec('outputs/round113/seal.json'),'plan':rec(EXTRA_PLAN),'source_sha256':source,'candidates':{},
          'query_labels_used_for_fit':False,'Public_score':None,'Private_score':None,
          'selected_submission':'v22','new_CSV':False},
          EXTRA_GROUP+'/preflight_review.json':pre,EXTRA_REVIEW:post}
    for n in (107,111,112):
        directory=f'outputs/round113/R{n}';metadata=f'outputs/round113/models/R{n}/'+('metadata.json' if n==112 else 'training.json')
        models=[rec(f'outputs/round113/models/R{n}/model.pt')]
        data[directory+'/stage.json']={'candidate_round':n,'stage':'offset_2023_02_11','fit_rows':272117,
            'query_rows':107169,'query_labels_available':False,'model_metadata':rec(metadata),'sealed_models':models}
        evaluation={'sealed_stage':rec(directory+'/stage.json'),'gate':gate['candidate_gates'][str(n)],
                    'both_point_metrics_improve':True}
        data[directory+'/evaluation/results.json']=evaluation;data[EXTRA_RESULTS]['candidates'][str(n)]=evaluation
        data[metadata]={'status':'completed','round':n}
        post['artifacts'][str(n)]={'stage':rec(directory+'/stage.json'),'training_metadata':rec(metadata),
                                  'evaluation':rec(directory+'/evaluation/results.json'),'sealed_models':list(models)}
        post['trainmeta'][str(n)]=dict(data[metadata])
        post['manual_gate_verdicts'][str(n)]={**gate['candidate_gates'][str(n)],'both_point_metrics_improve':True}
    for short in ('training_execution','evaluation_execution'):data[EXTRA_GROUP+'/'+short+'.json']={'exit_code':0}
    class MissingPath:
        def is_file(self):return False
    with patch.dict(globals(),{'read':lambda name:data[name],'reference':rec,'safe':lambda root,name:MissingPath()}):
        extra_evidence();need(required_extra_origin(111)['candidate_gate']==gate['candidate_gates']['111'],
                              'Exact R113 nested evidence required')
        checks.append('R113_exact_gate_plan_results_and_review_binding')
        for key,bad in [('passed',False),('components',{'overall_regression_at_most_0_25pct':True})]:
            old=gate['candidate_gates']['111'][key];gate['candidate_gates']['111'][key]=bad
            reject('R113_inconsistent_gate_rejected',extra_evidence);gate['candidate_gates']['111'][key]=old
        old=post['results'];post['results']={'path':EXTRA_RESULTS,'sha256':'0'*64}
        reject('R113_review_results_mismatch',extra_evidence);post['results']=old
        old=post['source_sha256'];post['source_sha256']=source
        reject('R113_missing_plan_source_binding',extra_evidence);post['source_sha256']=old
        old=data[EXTRA_RESULTS]['seal'];data[EXTRA_RESULTS]['seal']={'path':'outputs/round113/seal.json','sha256':'0'*64}
        reject('R113_result_seal_mismatch',extra_evidence);data[EXTRA_RESULTS]['seal']=old
        for key,bad in (('request',{'path':'outputs/round113/request.json','sha256':'0'*64}),
                        ('seal',{'path':'outputs/round113/seal.json','sha256':'0'*64}),
                        ('training_execution',{'path':EXTRA_GROUP+'/training_execution.json','sha256':'0'*64}),
                        ('evaluation_execution',{'path':EXTRA_GROUP+'/evaluation_execution.json','sha256':'0'*64}),
                        ('protected_unchanged',False),('new_model_fits',1),('GPU_work',True),
                        ('Private_score',0),('Public_score',0),('selected_submission','v28'),('new_CSV',True),
                        ('blocking_findings_remaining',['unresolved'])):
            old=post[key];post[key]=bad
            reject('R113_exact_postfit_'+key,extra_evidence);post[key]=old
        old=post['artifacts']['107'];post['artifacts'].pop('107')
        reject('R113_missing_reviewed_candidate',extra_evidence);post['artifacts']['107']=old
        old=post['artifacts']['111']['sealed_models'];post['artifacts']['111']['sealed_models']=[]
        reject('R113_model_review_subset_rejected',extra_evidence);post['artifacts']['111']['sealed_models']=old
        old=post['trainmeta']['112'];post['trainmeta']['112']={'status':'started'}
        reject('R113_full_native_metadata_mismatch',extra_evidence);post['trainmeta']['112']=old
        old=post['manual_gate_verdicts']['111'];post['manual_gate_verdicts']['111']={**old,'passed':False}
        reject('R113_manual_gate_mismatch',extra_evidence);post['manual_gate_verdicts']['111']=old
        post['manual_gate_verdicts']['111']={**old,'components':{k:1 for k in old['components']}}
        reject('R113_manual_components_must_be_booleans',extra_evidence);post['manual_gate_verdicts']['111']=old
        post['manual_gate_verdicts']['111']={**old,'both_point_metrics_improve':False}
        reject('R113_manual_point_metrics_mismatch',extra_evidence);post['manual_gate_verdicts']['111']=old
        gate['candidate_gates']['111']['passed']=False
        gate['candidate_gates']['111']['components']['overall_regression_at_most_0_25pct']=False
        post['manual_gate_verdicts']['111']['passed']=False
        reject('R113_failed_gate_blocks_final_release',lambda:required_extra_origin(111))
    class SidecarPath:
        name='receipt.json'
        def with_suffix(self,suffix):return self
        def is_file(self):return True
    with patch.dict(globals(),{'safe':lambda root,name:SidecarPath(),'sha':lambda path:'a'*64}):
        check_sidecar('outputs/round111/receipt.sha256',('a'*64+'\n').encode('ascii'))
        check_sidecar('outputs/round111/receipt.sha256',('a'*64+'  receipt.json\n').encode('ascii'))
        checks.append('bare_and_exact_same_directory_JSON_sidecar_allowed')
        for filename in ('../receipt.json','other.json','receipt.json\n'+('b'*64)+'  other.json'):
            reject('sidecar_other_path_or_multifile_rejected',lambda filename=filename:
                   check_sidecar('outputs/round111/receipt.sha256',('a'*64+'  '+filename+'\n').encode('ascii')))
    plan='reports/round111_final_plan.md';pre_name=GROUP+'/round111_final_release_preflight.json'
    sources={'src/round111_112_full_release.py':rec('src/round111_112_full_release.py')['sha256'],
             EXACT_BUILDER:rec(EXACT_BUILDER)['sha256']}
    final_req={'status':'frozen_before_one_exploratory_final_fit','round':111,'version':28,'new_fits':1,
               'query_labels_available':False,'selected_submission_preserved':'v22','Public_score':None,
               'Private_score':None,'sources':sources,'paths':{'plan':plan,'preflight':pre_name},
               'inputs':{plan:rec(plan)['sha256'],pre_name:rec(pre_name)['sha256']}}
    final_review={'source_sha256':dict(sources)}
    final_pre={'status':'passed','round':111,'version':28,'source_sha256':{**sources,plan:rec(plan)['sha256']}}
    with patch.dict(globals(),{'read':lambda name:final_pre,'reference':rec}):
        need(EXACT_BUILDER in check_final_source_binding(111,final_req,final_review),'Complete final closure allows exact old builder')
        checks.append('final_frozen_source_closure_and_separate_preflight_plan')
        old=final_review['source_sha256'];final_review['source_sha256']={EXACT_BUILDER:sources[EXACT_BUILDER]}
        reject('final_review_partial_source_closure',lambda:check_final_source_binding(111,final_req,final_review))
        final_review['source_sha256']=old
        old=final_pre['source_sha256'];final_pre['source_sha256']=dict(sources)
        reject('final_preflight_missing_plan_binding',lambda:check_final_source_binding(111,final_req,final_review))
        final_pre['source_sha256']=old
    class FakePath:
        def __truediv__(self,name):return self
        def exists(self):return False
        def glob(self,name):return []
    with patch.dict(globals(),{'ROOT':FakePath(),'safe':lambda root,name:FakePath()}):
        status=check_disposition(111,{'no_release_reason':'R113 fixed extra origin failed'},
                                {'gates':{'exploratory_submission':{'passed':True}}},False)
        need(status=='extra_gate_failed_no_release','Do not mislabel passed original gate as failed')
        checks.append('extra_gate_failed_no_release_distinct_from_original_failed')
    return {'status':'passed','checks':checks,'check_count':len(checks),'filesystem_writes':False,
            'file_copies':0,'new_model_fits':0,'GPU_calls':0,'git_commands':0,'network':False,
            'implementation_parent':IMPLEMENTATION_PARENT}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read-only full completion and packaging preflight (default)')
    mode.add_argument('--apply', action='store_true', help='After review, copy exact new evidence and refresh top manifest')
    mode.add_argument('--refresh-manifest', action='store_true', help='Separate finish after dated navigation notes')
    mode.add_argument('--self-check', action='store_true', help='Pure in-memory exclusion-policy checks')
    parser.add_argument('--release', action='append', default=[], metavar='ROUND=PATH')
    parser.add_argument('--no-release', action='append', default=[], metavar='ROUND=REASON')
    parser.add_argument('--defer-release', action='append', default=[], metavar='ROUND=REASON',
                        help='Explicitly defer a passed exploration gate before full-data release; concrete reason required')
    parser.add_argument('--release-review', action='append', default=[], metavar='ROUND=PATH',
                        help='Passed exact-bound independent final release review JSON; required for every released round')
    parser.add_argument('--extra-file', action='append', default=[], metavar='RELATIVE_PATH')
    arguments = parser.parse_args()
    if arguments.self_check:
        result = self_check()
    elif arguments.refresh_manifest:
        result = finish_manifest()
    else:
        result = run(parse_dispositions(arguments.release, arguments.no_release, arguments.defer_release),
                     arguments.extra_file, arguments.apply, parse_release_reviews(arguments.release_review))
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
