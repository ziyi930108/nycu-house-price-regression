"""Bounded R107--R110 source/evidence packaging; standard library only.

Default --check is read-only. After independent review, --apply copies exact
source/evidence bytes and writes a dated summary, independent copy manifest and
refreshed top-level SOURCE_MANIFEST. It never imports ML code, fits, loads private
models/arrays, runs git, uploads, or changes the selected Kaggle submission.

R107 release defaults to outputs/round107/final/release.json. Each other round
needs an explicit --release ROUND=PATH or --no-release ROUND=REASON. Missing
required completion artifacts fail before copying. Add a later v26 with e.g.
--release 110=outputs/round110/final/release.json. --extra-file adds a named,
policy-checked report/source/aggregate artifact; no broad outputs traversal.

Navigation is a separate finish step: prepend/append a dated note linking the
new docs page to README.md, research/README.md and/or HANDOFF.md. Keep the old
bodies exact, then run --refresh-manifest. Original navigation and SOURCE_MANIFEST
bytes are archived; historical 379-source manifest, production 13 sources and
all other pre-existing package files remain protected. No git command is run.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / 'deliverables/v22_github'
DATE = '20261008'
ROUNDS = (107, 108, 109, 110)
STAGES = ('development', 'confirmation', 'auxiliary')
GROUPS = ('outputs/research_resume_20261008', 'outputs/research_round109_110_20261008')
SNAPSHOT = 'evidence/outputs/research_package_resume_20261008'
COPY_MANIFEST = SNAPSHOT + '/copy_manifest.json'
SUMMARY = 'docs/RESEARCH_RESUME_20261008.md'
NAVIGATION = ('README.md', 'research/README.md', 'HANDOFF.md')
ALLOWED_SUFFIXES = {'.py', '.mjs', '.md', '.json', '.sha256', '.log'}
FORBIDDEN_PARTS = {'node_modules', '__pycache__', '.git', '.venv', 'venv',
                   'research_csv_intermediate_20261008', 'rawinput', 'raw_input', 'rawinputs', 'frames'}
FORBIDDEN_NAMES = {'feature_state.json', 'state.json', 'state_summary.json', 'price_prior.json',
                   'prior_state.json', 'encoder_state.json', 'v25_rows.json', 'v26_rows.json',
                   'csv_authoring_receipt.json'}
SECRET = re.compile(r'(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|sk-(?:proj-)?[A-Za-z0-9_-]{24,})')
SOURCE_FILES = (
    'src/round107_108_research.py', 'src/round107_final_candidate.py',
    'src/round107_full_release.py', 'src/round107_weighted_smooth_train.py',
    'src/round108_single_ctr_unit.py', 'src/round109_110_research.py',
    'src/round109_tail_parameter_average_train.py', 'src/round110_target_borders_unit.py',
    'src/package_research_resume_20261008.py',
)
REPORT_FILES = ('reports/round107_108_plan.md', 'reports/round107_108_report.md',
                'reports/round107_final_plan.md', 'reports/round109_110_plan.md',
                'reports/round109_110_report.md')
PRODUCTION = (
    'src/house_features.py', 'src/nn_predict.py', 'src/nn_prepare.py', 'src/predict_v22.py',
    'src/round47_nn_encode.py', 'src/round54_numeric_encode.py', 'src/submission_io.py', 'src/tree_runtime.py',
    'src/closure_modules/model_common.py', 'src/closure_modules/prepare_data.py',
    'src/closure_modules/round2_neighborhood.py', 'src/closure_modules/round6_geo.py',
    'src/closure_modules/round8_spatial_reference.py',
)
METHODS = {
    107: '日期加權 prior NN：固定10萬元 SmoothL1，減少微小殘差附近梯度跳動',
    108: '單價 CatBoost：組合 CTR 複雜度2→1，檢查降低類別交互容量是否有益',
    109: '原 R63 loss：固定 epoch21–30 參數等權平均，檢查尾段波動；非標準SWA',
    110: '原 R79 單價：僅 global CTR target borders=8，檢查更多價格分布統計',
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
    path = PurePosixPath(name)
    need(name == path.as_posix() and not path.is_absolute() and '..' not in path.parts,
         'Unsafe/noncanonical relative path: ' + name)
    need(not any(part.lower() in FORBIDDEN_PARTS for part in path.parts), 'Excluded directory: ' + name)
    return path


def safe(root, name):
    relative(name)
    path = (root / name).resolve()
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


def check_aggregate(value, key='$'):
    """Allow references/counts/metrics, reject embedded rows or numeric vectors."""
    need(not (key.lower() in {'inspection', 'inspection_ndjson', 'preview_rows', 'raw_preview',
                             'serialized_rows', 'csv_text', 'csv_content', 'row_values'}
              and isinstance(value, (dict, list, str))), 'Embedded inspection/row payload is excluded: ' + key)
    if isinstance(value, dict):
        for child, item in value.items():
            check_aggregate(item, child)
    elif isinstance(value, list):
        lowered = key.lower()
        need(lowered not in {'rows', 'csv_rows', 'raw_rows', 'frame_rows', 'data_rows',
                             'prices', 'predictions', 'targets', 'labels', 'ids', 'values', 'sample_values', '編號', '總價元'},
             'Embedded per-row data is not package evidence: ' + key)
        need(not (len(value) >= 256 and all(isinstance(item, (int, float)) for item in value)),
             'Long numeric vector is not aggregate evidence: ' + key)
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
        need(len(path.parts) == 2 and re.match(r'round(?:107|108|109|110)(?:_|\.)', path.name),
             'Only R107--R110 reports are in scope')
    else:
        allowed_root = '/'.join(path.parts[:2]) in GROUPS or bool(re.fullmatch(r'round(?:107|108|109|110)', path.parts[1]))
        need(allowed_root and len(path.parts) >= 3, 'Output outside the bounded round/request roots')
    if payload is not None:
        need(len(payload) <= 32 * 1024 * 1024, 'Unexpectedly large evidence file: ' + name)
        text = payload.decode('utf-8-sig')
        need(SECRET.search(text) is None, 'Possible credential in selected evidence; refusing copy: ' + name)
        if path.suffix == '.json':
            check_aggregate(json_value(payload))
        elif path.suffix == '.log':
            for line in text.splitlines():
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


def direct_files(directory):
    """One known directory only: no rglob, private-model traversal or node_modules."""
    path = safe(ROOT, directory)
    need(path.is_dir(), 'Required evidence directory missing: ' + directory)
    return [p.relative_to(ROOT).as_posix() for p in sorted(path.iterdir())
            if p.is_file() and p.suffix.lower() in ALLOWED_SUFFIXES and p.name.lower() not in FORBIDDEN_NAMES]


def parse_dispositions(releases, no_releases):
    selected = {107: {'release': 'outputs/round107/final/release.json'}}
    explicit = set()
    for kind, values in (('release', releases), ('no_release_reason', no_releases)):
        for specification in values:
            left, separator, right = specification.partition('=')
            need(separator and left.isdigit() and int(left) in ROUNDS and right.strip(),
                 'Expected ROUND=PATH or ROUND=REASON for rounds 107--110')
            number = int(left)
            need(number not in explicit, 'A round disposition was supplied twice')
            explicit.add(number)
            if kind == 'release':
                path = policy(right)
                need(path.suffix == '.json' and path.parts[1] == 'round' + str(number)
                     and path.name == 'release.json', 'Release must name that round release.json')
            selected[number] = {kind: right.strip()}
    need(set(selected) == set(ROUNDS), 'Explicit --release or --no-release required for rounds 108,109,110')
    return selected


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


def inventory(dispositions, extras):
    names = set(SOURCE_FILES) | set(REPORT_FILES)
    projections = []
    for first, group in enumerate(GROUPS):
        core = ('request.json', 'request.sha256', 'seal.json', 'seal.sha256', 'preflight_review.json',
                'postfit_review.json', 'training_execution.json', 'training_execution.sha256',
                'evaluation_execution.json', 'evaluation_execution.sha256')
        names.update(group + '/' + filename for filename in core)
        checks = ('R107_loss_selfcheck_execution.json', 'R108_helper_selfcheck_execution.json', 'user_request.json') if first == 0 else (
            'R109_average_selfcheck_execution.json', 'R110_helper_selfcheck_execution.json', 'ctr_review.json')
        names.update(group + '/' + filename for filename in checks)
        names.update(direct_files(group))
        request = read(group + '/request.json')
        need(request['status'] == 'frozen_before_six_fits', 'Research request is not the preregistered six-fit request')
        need(request['plan']['sha256'] == sha(safe(ROOT, request['plan']['path'])), 'Frozen research plan changed')
        seal = read(group + '/seal.json')
        need(seal['status'] == 'six_models_all_query_predictions_sealed_unscored', 'Research seal is incomplete')
        need(seal['request'] == {'path': group + '/request.json', 'sha256': sha(safe(ROOT, group + '/request.json'))},
             'Research seal/request identity mismatch')
        for receipt in ('preflight_review.json', 'postfit_review.json'):
            review = read(group + '/' + receipt)
            need(review['status'] == 'passed', 'Review has not passed: ' + group + '/' + receipt)
            for source, expected in review.get('source_sha256', {}).items():
                need(source.startswith(('src/', 'reports/')) and sha(safe(ROOT, source)) == expected,
                     'Reviewed source/plan changed: ' + source)
        for receipt in ('training_execution.json', 'evaluation_execution.json'):
            need(read(group + '/' + receipt)['exit_code'] == 0, 'Research execution has not completed: ' + receipt)
        for source, expected in request['source_sha256'].items():
            need(source.startswith('src/') and sha(safe(ROOT, source)) == expected, 'Frozen research source changed: ' + source)
            names.add(source)
    for number in ROUNDS:
        directory = 'outputs/round' + str(number)
        names.update((directory + '/results.json', directory + '/results.sha256'))
        results = read(directory + '/results.json')
        need(results['status'] == 'three_reused_windows_evaluated' and set(results['stages']) == set(STAGES),
             'Round has not finished all three windows: ' + str(number))
        need(results.get('Private_score') is None, 'Unexpected Private score; update the packaging contract explicitly')
        group = GROUPS[0 if number in (107, 108) else 1]
        for field in ('request', 'seal'):
            name = group + '/' + field + '.json'
            need(results[field] == {'path': name, 'sha256': sha(safe(ROOT, name))}, 'Results reference a different request/seal')
        seal = read(group + '/seal.json')
        for stage in STAGES:
            folder = directory + '/' + stage
            names.update(folder + '/' + filename for filename in ('stage.json', 'stage.sha256', 'train.log'))
            names.update(direct_files(folder))
            names.update((folder + '/evaluation/results.json', folder + '/evaluation/results.sha256'))
            names.update(direct_files(folder + '/evaluation'))
            model_files = ('run.json', 'training.json') if number in (107, 109) else ('started.json', 'metadata.json')
            if number == 110:
                model_files += ('target_border_summary.json',)
            names.update(folder + '/model/' + filename for filename in model_files)
            names.update(direct_files(folder + '/model'))
            stage_record = read(folder + '/stage.json')
            need(seal['records'][str(number)][stage] == {
                'path': folder + '/stage.json', 'sha256': sha(safe(ROOT, folder + '/stage.json'))},
                 'Stage differs from the original seal')
            need(read(folder + '/evaluation/results.json') == results['stages'][stage], 'Stage and aggregate evaluation differ')
            training = read(folder + '/model/' + ('training.json' if number in (107, 109) else 'metadata.json'))
            metadata_name = folder + '/model/' + ('training.json' if number in (107, 109) else 'metadata.json')
            need(stage_record['model_metadata'] == {'path': metadata_name, 'sha256': sha(safe(ROOT, metadata_name))},
                 'Stage metadata identity differs')
            need(training.get('saved_model_reload_exact') is True, 'Native reload verification not complete')
            if number in (107, 109):
                need(training['status'] == 'completed', 'NN training has not completed')
            if number == 109:
                need(training.get('saved_epoch30_model_reload_exact') is True, 'R109 same-trajectory control is not verified')
        disposition = dispositions[number]
        if 'release' in disposition:
            release_name = disposition['release']
            released = read(release_name)
            need(released['status'] == 'exploratory_submission_ready_not_selected'
                 and released['round'] == number and released['rows'] == 47073,
                 'Final release is not a completed exploratory submission')
            need(results['gates']['exploratory_submission']['passed'] is True,
                 'A final release is not supported by the fixed exploration gate')
            need(released.get('Public_score') is None and released.get('Private_score') is None,
                 'New score feedback needs a separately reviewed dated evidence update')
            need(released['selected_submission_preserved'] == 'v22'
                 and released['automatic_Kaggle_upload_or_selection'] is False,
                 'Release must preserve v22 and avoid automatic Kaggle action')
            folder = str(PurePosixPath(release_name).parent)
            names.update(folder + '/' + filename for filename in (
                'request.json', 'request.sha256', 'seal.json', 'seal.sha256', 'release.json', 'release.sha256',
                'fit_execution.json', 'fit_execution.sha256',
                'csv_authoring.log', 'train.log'))
            names.update(direct_files(folder))
            names.update(direct_files(folder + '/model'))
            final_model_files = ('run.json', 'training.json') if number in (107, 109) else ('started.json', 'metadata.json')
            if number == 110:
                final_model_files += ('target_border_summary.json',)
            names.update(folder + '/model/' + filename for filename in final_model_files)
            final_metadata = read(folder + '/model/' + ('training.json' if number in (107, 109) else 'metadata.json'))
            need(final_metadata.get('saved_model_reload_exact') is True
                 and final_metadata['fit_rows'] == 467221 and final_metadata['query_rows'] == 47073,
                 'Full-data native metadata/reload verification is incomplete')
            need(read(folder + '/fit_execution.json')['exit_code'] == 0, 'Full-data fit execution did not complete')
            for field in ('request', 'seal'):
                name = folder + '/' + field + '.json'
                need(released[field] == {'path': name, 'sha256': sha(safe(ROOT, name))}, 'Release request/seal identity differs')
            # The original authoring receipt embeds preview ID/prices in its
            # inspection NDJSON. Read only this known receipt, keep its hash and
            # publish a clearly declared aggregate projection, never its rows.
            authoring_name = folder + '/csv_authoring_receipt.json'
            authoring_bytes = safe(ROOT, authoring_name).read_bytes()
            authoring = json_value(authoring_bytes)
            need(released['authoring'] == {'path': authoring_name, 'sha256': sha_bytes(authoring_bytes)},
                 'Release authoring-receipt identity differs')
            projection = authoring_aggregate(authoring, authoring_name, sha_bytes(authoring_bytes), released['CSV']['sha256'])
            projections.append({'source': authoring_name, 'source_sha256': sha_bytes(authoring_bytes),
                                'destination': 'evidence/' + folder + '/csv_authoring_aggregate.json',
                                'payload': encoded(projection)})
        else:
            need(len(disposition['no_release_reason']) >= 8, 'Supply a concrete non-release reason')
    for pattern in ('round107_*', 'round108_*', 'round109_*', 'round110_*'):
        names.update(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'reports').glob(pattern))
                     if p.is_file() and p.suffix in {'.md', '.json', '.sha256'})
        names.update(p.relative_to(ROOT).as_posix() for p in sorted((ROOT / 'src').glob(pattern + '.py')) if p.is_file())
    names.update(extras)
    missing = [name for name in sorted(names) if not safe(ROOT, name).is_file()]
    need(not missing, 'Missing required completion evidence; main thread must resolve status:\n' + '\n'.join(missing))
    entries = []
    for name in sorted(names):
        payload = safe(ROOT, name).read_bytes()
        policy(name, payload)
        if name.endswith('.sha256'):
            counterpart = safe(ROOT, name).with_suffix('.json')
            if counterpart.is_file():
                need(payload.decode('ascii').strip() == sha(counterpart), 'SHA sidecar differs: ' + name)
        entries.append({'source': name, 'destination': destination(name), 'bytes': len(payload), 'sha256': sha_bytes(payload)})
    return entries, projections


def package_files(additions=(), template=None):
    """Hash only old manifest members plus explicit additions, never private/runs.

    The local checkout legitimately contains ignored private models and replay
    outputs. Filesystem-wide discovery would publish those in SOURCE_MANIFEST.
    Its old explicit list, rather than os.walk/rglob or git, defines the scope.
    """
    if template is None:
        template = json_value((REPO / 'SOURCE_MANIFEST.json').read_bytes())
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
    old = json_value((REPO / 'SOURCE_MANIFEST.json').read_bytes())
    expected = {item['path']: item for item in old['files']}
    for name in PRODUCTION + ('research/source_manifest.json',):
        need(name in records and records[name] == expected.get(name), 'Protected original source differs: ' + name)
    history = json_value((REPO / 'research/source_manifest.json').read_bytes())
    need(isinstance(history, list) and len(history) == 379, 'Historical 379-source manifest changed')
    for item in history:
        need(records[item['copy']]['sha256'] == item['sha256'], 'Historical source copy changed: ' + item['copy'])
    return old, {name: item['sha256'] for name, item in records.items()
                 if name != 'SOURCE_MANIFEST.json' and name not in NAVIGATION}


def summary(dispositions):
    lines = ['# 2026-10-08 恢復研究：R107–R110', '',
             '使用者重新授權改良模型及新試交。v22與原簡報保留；v22已回報Public MAE為1,998,568.5706元，Private未知。',
             '以下為反覆使用歷史三窗的離線MAE；新候選Public／Private均未知，沒有代替使用者上傳或勾選Kaggle。', '',
             '|方法與固定理由|development ΔMAE|confirmation ΔMAE|auxiliary ΔMAE|探索gate|強gate|產物終態|',
             '|---|---:|---:|---:|---|---|---|']
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
            role = '未發行：' + status['no_release_reason'].replace('|', '／').replace('\n', ' ')
        link = f'../evidence/outputs/round{number}/results.json'
        lines.append(f"|[R{number}]({link})：{METHODS[number]}|" + '|'.join(deltas)
                     + f"|{gates['exploratory_submission']['passed']}|{gates['promotion']['adoption_research_gate_passed']}|{role}|")
    lines.extend(['', 'ΔMAE單位為新臺幣元，負值表示優於保存v22。完整整體／新舊加權、分群、五層、v24反證與配對區間在原results及報告；本頁不省略失敗而改稱所有方法成功。', '',
        '探索gate允許每窗指標至多0.25%退步，要求最新兩指標不退步、至少兩窗整體改善且無季度／大型群超5%退步；強gate要求三窗整體及加權均改善、最新整體至少0.5%，同樣限制分群退步。通過探索不等於強推薦或Private較佳。', '',
        '流程固定先提出假設與切點，再核canonical索引／九鍵NN輸入／fit-only權重與OOF價格先驗、各切點狀態，序列fit、封存模型與query向量，之後才讀query真價評分。各研究獨立，未把不同輪變因合併，也未依Public搜尋參數。', '',
        '新提交格式：47,073列，欄名`編號,總價元`；依原sample ID順序，總價於最後一次取整並至少1元。程式／證據包僅保留提交SHA及格式收據，沒有CSV、逐筆價格、NPZ／NPY、模型、encoder／prior／reference狀態或課程原表。', '',
        '限制：重用驗證窗口仍有累積選型偏差；條件bootstrap不是Private勝率；GPU固定seed不保證重訓逐bytes相同。來源可追溯及保存重載驗證，不等於新電腦從零端到端重訓已完成。未到內政部成交系統對答案。', '',
        f'[本輪精確複製清單](../{COPY_MANIFEST})；原379歷史manifest及所有舊收據保持。README／research README／HANDOFF的舊「停止研究」說法屬其原日期，請以本頁及本輪request／release為新增現況。', ''])
    return '\n'.join(lines).encode('utf-8')


def write_new(path, payload):
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
    replacement['research_resume_20261008'] = {'copy_manifest': COPY_MANIFEST, 'publication_not_performed': True}
    target = REPO / 'SOURCE_MANIFEST.json'
    temporary = target.with_name('SOURCE_MANIFEST.resume_20261008.tmp')
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


def run(dispositions, extras, apply):
    need(REPO.is_dir(), 'Existing v22 GitHub package is required')
    entries, projections = inventory(dispositions, extras)
    current = package_files()
    template, protected = protect_existing(current)
    need(not (REPO / COPY_MANIFEST).exists() and not (REPO / SUMMARY).exists(),
         'Preserve prior packaging; do not overwrite a completed snapshot')
    document = summary(dispositions)
    for item in entries:
        need(not (REPO / item['destination']).exists() or item['destination'] in current,
             'Unlisted destination already exists; main thread must resolve its status: ' + item['destination'])
        if item['destination'] in current:
            need(current[item['destination']]['sha256'] == item['sha256'], 'Existing research/evidence copy differs: ' + item['destination'])
    need(all(not (REPO / item['destination']).exists() for item in projections), 'Preserve any prior aggregate projection')
    planned = {'status': 'preflight_passed_no_writes' if not apply else 'ready_to_copy',
               'files': len(entries), 'bytes': sum(item['bytes'] for item in entries),
               'aggregate_receipt_projections': len(projections),
               'round_dispositions': dispositions, 'model_fits': 0, 'git_or_network': False}
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
               'generated': generated, 'round_dispositions': dispositions, 'protected_before': protected,
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
    receipt = json_value((REPO / COPY_MANIFEST).read_bytes())
    verify_protected(receipt['protected_before'])
    for item in receipt['files']:
        need(sha(safe(REPO, item['destination'])) == item['sha256'], 'Packaged source/evidence changed')
    for item in receipt['generated']:
        need(sha(safe(REPO, item['path'])) == item['sha256'], 'Generated document/projection/archive changed')
    for name in NAVIGATION:
        original = (REPO / SNAPSHOT / 'before' / name).read_bytes()
        current = (REPO / name).read_bytes()
        need(current.startswith(original) or current.endswith(original), 'Navigation must preserve original body bytes: ' + name)
    finish_path = REPO / SNAPSHOT / 'finish_manifest_receipt.json'
    need(not finish_path.exists(), 'Preserve an existing finish receipt')
    finish = {'status': 'dated_navigation_finish_local_only', 'created_at_utc': datetime.now(timezone.utc).isoformat(),
              'navigation': {name: sha(REPO / name) for name in NAVIGATION},
              'copy_manifest_sha256': sha(REPO / COPY_MANIFEST), 'protected_files_unchanged': True,
              'git_or_network': False}
    write_new(finish_path, encoded(finish))
    return {**finish, 'SOURCE_MANIFEST': refresh_top(
        json_value((REPO / 'SOURCE_MANIFEST.json').read_bytes()), [finish_path.relative_to(REPO).as_posix()])}


def self_check():
    """Pure in-memory policy checks; no files, copies, git, ML import or fitting."""
    check_aggregate({'rows': 47073, 'CSV': {'path': 'outputs/submission_v25_trial.csv', 'sha256': 'a' * 64},
                     'epochs': [{'epoch': n} for n in range(30)], 'cardinalities': list(range(23))})
    rejected = []
    for name in ('outputs/round107/model.pt', 'outputs/round110/model.cbm', 'outputs/round107/feature_state.json',
                 'outputs/research_csv_intermediate_20261008/v25_rows.json', 'outputs/round107/a.npy',
                 'outputs/round107/node_modules/x.json', '../outside.json', 'C:/outside.json'):
        try:
            policy(name)
        except ValueError:
            rejected.append(name)
        else:
            raise AssertionError('Forbidden path accepted: ' + name)
    for value in ({'rows': [[1, 100]]}, {'predictions': [1, 2]}, {'values': list(range(256))},
                  {'inspection': '{"values":[["123",1000000]]}'}):
        try:
            check_aggregate(value)
        except ValueError:
            rejected.append('embedded_rows_or_vector')
        else:
            raise AssertionError('Embedded row data accepted')
    selected = parse_dispositions(['110=outputs/round110/final/release.json'], ['108=failed fixed gates', '109=not released after review'])
    need(selected[107]['release'].endswith('release.json') and selected[110]['release'].startswith('outputs/round110/'), 'Release extension failed')
    authoring = {'status': 'artifact_typed_range_exported_to_csv', 'rows': 47073, 'columns': ['編號', '總價元'],
                 'input_json_sha256': 'a' * 64, 'csv_sha256': 'b' * 64, 'full_workbook_values_exact': True,
                 'formula_count': 0, 'utf8_bom': True, 'csv_encoding': 'comma-delimited CRLF',
                 'inspection': '{"values":[["private-ID",999999999]]}'}
    aggregate = authoring_aggregate(authoring, 'outputs/round107/final/csv_authoring_receipt.json', 'c' * 64, 'b' * 64)
    need('inspection' not in aggregate and b'private-ID' not in encoded(aggregate)
         and 'inspection' in authoring, 'Authoring projection must omit rows without altering its source')
    return {'status': 'passed', 'policy_rejections': len(rejected), 'optional_v26_round_release_supported': True,
            'authoring_inspection_payload_omitted_by_allowlist': True,
            'file_copies': 0, 'model_fits': 0, 'git_or_network': False, 'filesystem_writes': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check', action='store_true', help='Read-only full completion and packaging preflight (default)')
    mode.add_argument('--apply', action='store_true', help='After review, copy exact new evidence and refresh top manifest')
    mode.add_argument('--refresh-manifest', action='store_true', help='Separate finish after dated navigation notes')
    mode.add_argument('--self-check', action='store_true', help='Pure in-memory exclusion-policy checks')
    parser.add_argument('--release', action='append', default=[], metavar='ROUND=PATH')
    parser.add_argument('--no-release', action='append', default=[], metavar='ROUND=REASON')
    parser.add_argument('--extra-file', action='append', default=[], metavar='RELATIVE_PATH')
    arguments = parser.parse_args()
    if arguments.self_check:
        result = self_check()
    elif arguments.refresh_manifest:
        result = finish_manifest()
    else:
        result = run(parse_dispositions(arguments.release, arguments.no_release), arguments.extra_file, arguments.apply)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, allow_nan=False))
