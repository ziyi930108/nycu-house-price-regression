"""Freeze an accepted, completed R6 release as runs/v3; never train or score.

Default: create an immutable snapshot. --verify: read-only integrity audit.
--register: verify an existing snapshot and release, back up the registry, then
append v3 without replacing any previous entry. No upload is performed.
"""
from __future__ import annotations
import argparse, ast, csv, hashlib, json, os, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'outputs/round6'
DEST = ROOT/'runs/v3'
MANIFEST, RECEIPT, NOTICE_FILE = 'snapshot_manifest.json', 'snapshot_manifest.sha256', 'SNAPSHOT_README.md'
V2_SHA = 'ef44b7f520bbd15a5dd134f7f24f9c13729ddd6d0868f025bc156af51166654b'
R5_SHA = '10444c2e1be03f2e3a6354784c4b6cb9bf5baf8dcf2b5b4e99026c5c9f75d4bf'
REFERENCE = 'outputs/round2/neighborhood/final_reference.pkl'
REQUIRED_REPORTS = ('README_round6.md', 'reports/optimization_round6_report.md',
 'reports/round6_results.json', 'reports/round6_work_log.md',
 'reports/round6_preprocessing_record.md', 'reports/round6_final_checks.json')
SUFFIXES = {'.json', '.sha256', '.csv', '.npy', '.pkl', '.cbm', '.txt', '.md', '.py', '.png', '.svg', '.pdf', '.html', '.ipynb'}
OMIT_DIRS = {'catboost_info', '__pycache__', 'logs', 'log', 'notebook_runtime', 'matplotlib_cache'}
NOTICE = '''# v3: accepted round-six geographic release

This snapshot preserves the accepted fixed R6 candidate, all four stages of
encoded geography, current-round models and predictions, receipts, source and
final reporting. It is not a self-contained runtime: original spreadsheets,
feature caches, large OSM/road/matched data and older baseline models remain
exact-SHA external_dependencies. Keep those paths or restore their verified
previous backups. The entire geography or runs tree was not copied.

The historical independent_preflight_draft.json predates the prepare-source
provenance fix. Its old source was not preserved; this draft is evidence only,
not current authority or a claimed replayable historical implementation. Current
contracts are the frozen plan, final QA, prepared manifests and evaluation.

The manifest excludes itself and has a separate SHA receipt. Originals and
copies are hashed and sized individually. Failed staging is retained. --register
is a separate action and preserves a byte-for-byte pre-update registry backup.
The old v2 CSV and previous snapshots remain unchanged. This does not upload to
Kaggle or claim an observed Public/Private result for v3.
'''


def need(ok, message):
    if not ok: raise RuntimeError(message)


def safe(base, name):
    base = Path(base).resolve(); path = (base/Path(name)).resolve()
    need(path != base and path.is_relative_to(base), 'Path escapes allowed directory: '+str(name))
    return path


def rel(path): return safe(ROOT, path).relative_to(ROOT.resolve()).as_posix()
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def encoded(value): return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)+'\n').encode('utf-8')


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for chunk in iter(lambda: handle.read(1024*1024), b''): value.update(chunk)
    return value.hexdigest()


def verify_file(base, name, digest, size=None):
    path = safe(base, name)
    need(path.is_file(), 'Missing file: '+str(path))
    need(size is None or path.stat().st_size == size, 'File size differs: '+str(path))
    need(sha(path) == digest, 'File SHA differs: '+str(path))
    return path


def locked(path):
    path = safe(ROOT, path); receipt = path.with_suffix('.sha256')
    need(path.is_file() and receipt.is_file(), 'Missing contract/receipt: '+str(path))
    need(receipt.read_text(encoding='ascii').strip() == sha(path), 'Receipt differs: '+str(path))
    return read(path)


def registry_collect():
    expected, reasons = {}, {}
    def add(name, digest=None, reason='explicit dependency'):
        path = safe(ROOT, name); key = rel(path)
        need(path.is_file(), 'Missing declared dependency: '+key)
        actual = expected[key] if key in expected else sha(path)
        need(digest is None or (isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest) and digest == actual), 'Recorded SHA differs: '+key)
        expected[key] = actual; reasons.setdefault(key, set()).add(reason)
        return actual
    def mapping(value, reason):
        need(isinstance(value, dict), 'Expected SHA mapping: '+reason)
        for name, digest in value.items(): add(name, digest, reason)
    def contract(path, reason):
        value = locked(path); add(path, reason=reason); add(Path(path).with_suffix('.sha256'), reason=reason+' receipt')
        return value
    return expected, reasons, add, mapping, contract


def validate_contracts(add, mapping, contract):
    plan = contract(OUT/'candidate_plan.json', 'frozen plan')
    frozen = contract(OUT/'frozen_candidate.json', 'frozen selection')
    accepted = contract(OUT/'accepted_candidate.json', 'final acceptance')
    need(plan['round'] == 6 and plan['control'] == 'v2_control' and set(plan['candidates']) == {'geography_blend'}, 'Wrong planned round/candidate')
    need(set(plan['components']) == {'geography_catboost'}, 'Wrong planned component')
    need(accepted['accepted'] is True, 'No accepted R6 candidate: refuse release archive')
    need(frozen['selected_candidate'] == accepted['selected_candidate'] == accepted['deployment_decision'] == 'geography_blend', 'Release selection differs')
    need(accepted['candidate_reselection_performed'] is False, 'Candidate was reselected')
    psha, fsha, asha = sha(OUT/'candidate_plan.json'), sha(OUT/'frozen_candidate.json'), sha(OUT/'accepted_candidate.json')
    need(frozen['plan_sha256'] == accepted['plan_sha256'] == psha and accepted['frozen_candidate_sha256'] == fsha, 'Plan/selection/acceptance identity differs')
    need(frozen['required_components'] == plan['required_components']['geography_blend'] == ['geography_catboost'], 'Frozen components differ')
    for key in ('candidate_formulas', 'thresholds', 'target_group_weights'):
        need(frozen[key] == plan[key], 'Frozen rule differs: '+key)
    need(set(frozen['splits']) == set(plan['splits']), 'Frozen split set differs')
    for stage, split in frozen['splits'].items():
        need(split['dates'] == plan['splits'][stage], 'Frozen split dates differ')
        for key in ('fit_rows', 'query_rows'):
            need(split[key] == plan['split_contract'][stage][key], 'Frozen split count differs')
        folder = OUT/'models'/stage/'geography_catboost'
        for role in ('fit', 'query'):
            add(folder/(role+'_indices.npy'), split[role+'_indices_npy_sha256'], 'frozen '+stage+' row order')
    for key in ('fit_source_sha256', 'evaluation_source_sha256', 'data_sha256', 'research_evidence_sha256', 'geography_transform_source_sha256'):
        mapping(plan[key], 'plan '+key)
    add('src/round6_plan.py', plan['plan_creator_sha256'], 'plan creator')
    for key in ('source_sha256', 'data_sha256', 'development_artifact_sha256'): mapping(frozen[key], 'freeze '+key)
    mapping(accepted['evidence_sha256'], 'accepted evidence')
    need(set(accepted['stage_decisions']) == {'confirmation', 'auxiliary'} and all(v['passed'] is True for v in accepted['stage_decisions'].values()), 'Both later-stage gates must pass')
    evaluations = {}
    for stage in ('development', 'confirmation', 'auxiliary'):
        path = OUT/'evaluation'/stage/'results.json'; result = contract(path, stage+' evaluation')
        need(result['stage'] == stage and result['plan_sha256'] == psha, 'Evaluation stage/plan differs')
        need(set(result['evaluated_candidates']) == {'v2_control', 'geography_blend'}, 'Unexpected evaluation candidates')
        for key in ('candidate_formulas', 'thresholds', 'target_group_weights'):
            need(result[key] == plan[key], stage+' evaluation rule differs: '+key)
        for key in ('source_sha256', 'data_sha256', 'input_artifact_sha256', 'output_artifact_sha256'): mapping(result[key], stage+' '+key)
        need(result['decision']['selected_candidate'] == 'geography_blend', 'Evaluation selected another candidate')
        if stage != 'development': need(result['decision'] == accepted['stage_decisions'][stage], 'Acceptance stage evidence differs')
        evaluations[stage] = result
    for stage in ('development', 'confirmation', 'auxiliary', 'final'):
        path = OUT/'geography'/stage/'manifest.json'; geo = contract(path, stage+' geography')
        need(geo['stage'] == stage and geo['snapshot_year'] == plan['geography']['years'][stage], 'Prepared geography stage/year differs')
        need(geo['geography_provenance_chain']['verified'] is True and geo['geographic_values_learned_from_targets'] is False, 'Geography provenance not certified')
        stamp = datetime.fromisoformat(geo['replication_timestamp'].replace('Z', '+00:00'))
        origin = datetime.fromisoformat(geo['forecast_origin_utc'].replace('Z', '+00:00'))
        need(stamp < origin and stamp.year == geo['snapshot_year'], 'Map is not the planned historical snapshot')
        for key in ('source_sha256', 'index_files_sha256'): mapping(geo[key], stage+' geography '+key)
        for item in geo['features'].values(): add(item['path'], item['sha256'], stage+' encoded geography')
    release = contract(OUT/'release/release.json', 'published release')
    need(release['release_version'] == 'v3' and release['experiment_round'] == 6 and release['candidate'] == 'geography_blend', 'Wrong release identity')
    need(release['rows'] == 47073 and release['training_rows'] == 467221 and release['columns'] == ['編號', '總價元'], 'Release rows/columns differ')
    need(release['path'] == 'outputs/submission_v3.csv', 'Unexpected submission path')
    for key, value in (('plan_sha256', psha), ('frozen_candidate_sha256', fsha), ('accepted_candidate_sha256', asha)):
        need(release[key] == value, 'Release contract differs: '+key)
    need(release['formula'] == plan['candidate_formulas']['geography_blend'], 'Release formula differs')
    for key in ('full_component_replayed_all_rows', 'v2_csv_reproduced_exactly', 'id_order_verified', 'positive_finite_integer_prices'):
        need(release[key] is True, 'Release proof failed: '+key)
    need(release['external_property_answers_used'] is False and release['v2_unchanged_sha256'] == V2_SHA, 'Release source/retained-v2 policy differs')
    add(release['path'], release['sha256'], 'published submission')
    mapping(release['full_component_sha256'], 'final model artifacts')
    mapping(release['baseline_component_sha256'], 'preserved baseline models')
    add(REFERENCE, release['reference_sha256'], 'preserved final reference')
    add('src/round6_release.py', release['release_source_sha256'], 'release generator')
    add(OUT/'release/predictions.npy', release['unrounded_prediction_sha256'], 'release predictions')
    need(release['geography_manifest_path'] == rel(OUT/'geography/final/manifest.json'), 'Final geography path differs')
    add(release['geography_manifest_path'], release['geography_manifest_sha256'], 'final geography identity')
    model_dir = OUT/'models/final/geography_catboost'; meta = read(model_dir/'metadata.json')
    required = {'metadata.json', 'model.cbm', 'feature_state.json', 'predictions.npy', 'fit_indices.npy', 'query_indices.npy'}
    need(required <= {Path(name).name for name in release['full_component_sha256']}, 'Incomplete final model artifacts')
    need({rel(p) for p in model_dir.iterdir() if p.is_file()} == set(release['full_component_sha256']), 'Final model inventory differs')
    need(all(safe(ROOT, name).parent == model_dir.resolve() for name in release['full_component_sha256']), 'Final model path differs')
    need(meta['plan_sha256'] == psha and meta['frozen_candidate_sha256'] == fsha and meta['parameters'] == plan['components']['geography_catboost']['parameters'], 'Final model plan/parameters differ')
    need(meta['iterations'] == 2999 and meta['feature_count'] == 95 and meta['categorical_feature_count'] == 23, 'Final model shape differs')
    need(meta['outer_fit_rows'] == 467221 and meta['query_rows'] == 47073 and meta['reload_64_rows_verified'] is True, 'Final model fit/query/replay proof differs')
    mapping(meta['fit_source_sha256'], 'full model fit sources')
    for name, field in (('model.cbm', 'model_sha256'), ('feature_state.json', 'feature_state_sha256'), ('predictions.npy', 'prediction_sha256')):
        add(model_dir/name, meta[field], 'model metadata '+field)
    need(meta['geography_manifest_sha256'] == release['geography_manifest_sha256'], 'Model geography differs')
    return plan, frozen, accepted, release, evaluations


def verify_prior(plan, add, contract):
    r5path = ROOT/'runs/round5_experiments'/MANIFEST
    need(plan['prior_round5_manifest_sha256'] == R5_SHA, 'Unexpected prior R5 manifest')
    add(r5path, R5_SHA, 'retained R5 manifest'); prior = contract(r5path, 'retained R5 snapshot')
    folder = r5path.parent
    for name, item in prior['copied_files'].items():
        need(item['original_sha256'] == item['backup_sha256'] and item['original_bytes'] == item['backup_bytes'], 'Prior R5 copy proof differs')
        verify_file(folder, name, item['backup_sha256'], item['backup_bytes'])
    inventory = {relpath.relative_to(folder).as_posix() for relpath in folder.rglob('*') if relpath.is_file()}
    need(inventory == set(prior['copied_files'])|{MANIFEST, RECEIPT, prior['dependency_notice']}, 'Prior R5 inventory differs')
    tree = ast.parse((folder/'src/archive_round5.py').read_text(encoding='utf-8-sig'))
    notice = [ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'NOTICE' for t in n.targets)]
    need(len(notice) == 1 and (folder/prior['dependency_notice']).read_bytes() == notice[0].encode('utf-8'), 'Prior R5 notice differs')
    add(folder/prior['dependency_notice'], reason='prior R5 notice')
    v2proof = prior['original_plan']['prior_archives_verified']['v2']
    add(v2proof['manifest_path'], v2proof['manifest_sha256'], 'preserved v2 snapshot')
    v2 = read(ROOT/v2proof['manifest_path']); need(v2['version'] == 'v2', 'Wrong retained release')
    for name, digest in v2['files_sha256'].items(): verify_file(ROOT/'runs/v2', name, digest)
    need(v2['files_sha256']['outputs/submission_v2.csv'] == V2_SHA, 'Prior v2 submission differs')
    add('outputs/submission_v2.csv', V2_SHA, 'preserved v2 submission')
    add('reports/run_manifest.json', reason='original input inventory')
    inputs = read(ROOT/'reports/run_manifest.json')['input_sha256']; paths = {}
    names = {'train':'train-v2.xlsx', 'valid':'valid-v2.xlsx', 'test':'test-reindex-test-v2.2.xlsx'}
    need(set(inputs) == set(names.values())|{'test-reindex-test-sample-submission-v2.csv'}, 'Original input inventory differs')
    need(plan['original_input_sha256'] == {k:inputs[v] for k,v in names.items()}, 'Plan original inputs differ')
    for name, digest in inputs.items():
        found = [r for r in prior['external_dependencies'] if Path(r['original_path']).name == name and r['sha256'] == digest]
        need(len(found) == 1, 'Original input path absent/ambiguous: '+name)
        paths[name] = found[0]['original_path']; add(paths[name], digest, 'original input')
    return {'round5_manifest_sha256':R5_SHA, 'round5_verified_files':len(prior['copied_files']),
            'v2_manifest_path':v2proof['manifest_path'], 'v2_manifest_sha256':v2proof['manifest_sha256'],
            'v2_verified_files':len(v2['files_sha256'])}, paths['test-reindex-test-sample-submission-v2.csv']


def check_csv(release, sample_path):
    with safe(ROOT, sample_path).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.DictReader(handle); need(reader.fieldnames == ['編號', '總價元'], 'Sample columns differ')
        ids = [row['編號'] for row in reader]
    need(ids == [str(i) for i in range(1, 47074)], 'Sample ID order/rows differ')
    count, minimum, maximum = 0, None, None
    with safe(ROOT, release['path']).open(encoding='utf-8-sig', newline='') as handle:
        reader = csv.reader(handle); need(next(reader) == ['編號', '總價元'], 'Submission columns differ')
        for row in reader:
            need(len(row) == 2 and count < len(ids) and row[0] == ids[count], 'Submission row shape/ID order differs')
            need(re.fullmatch('[0-9]+', row[1]) is not None and 0 < int(row[1]) < 2**63, 'Submission price is not a positive finite int64')
            price = int(row[1]); minimum = price if minimum is None else min(minimum, price)
            maximum = price if maximum is None else max(maximum, price); count += 1
    need(count == release['rows'] == 47073 and minimum == release['minimum'] and maximum == release['maximum'], 'Submission summary differs')
    return {'rows':count, 'columns':['編號', '總價元'], 'sample_id_order_verified':True,
            'positive_finite_integer_prices':True, 'sha256':sha(ROOT/release['path'])}


def validate_reports(plan, accepted, release, evaluations, mapping, add):
    for name in REQUIRED_REPORTS: add(name, reason='required final report/check')
    report = read(ROOT/'reports/round6_results.json'); checks = read(ROOT/'reports/round6_final_checks.json')
    need(report['status'] == 'released_v3' and report['recommendation'] == 'v3' and report['backtest_accepted'] is True, 'Reports do not describe a completed accepted v3 release')
    need(report['release'] == release and report['acceptance'] == accepted and report['recommended_submission'] == release['path'], 'Report release/acceptance differs')
    need(report['new_submission'] == {k:release[k] for k in ('path', 'sha256', 'rows')}, 'Report submission differs')
    need(report['candidate_formulas'] == plan['candidate_formulas'] and report['thresholds'] == plan['thresholds'], 'Report rules differ')
    for stage, result in evaluations.items():
        item = report['stages'][stage]
        need(item['status'] == 'evaluated' and item['metrics'] == result['metrics'] and item['decision'] == result['decision'], 'Report differs from saved evaluation: '+stage)
    mapping(report['receipt_verified_inputs'], 'reported verified input')
    add('src/round6_report.py', report['report_script_sha256'], 'report generator')
    need(checks['all_checks_passed'] is True, 'Final QA not passed')
    for key in ('plan_sha256', 'frozen_candidate_sha256', 'accepted_candidate_sha256'):
        need(checks[key] == release[key], 'Final QA contract differs: '+key)
    need(checks['release_sha256'] == sha(OUT/'release/release.json') and checks['submission_sha256'] == release['sha256'], 'Final QA refers to another release/CSV')
    if 'checked_docs_sha256' in checks: mapping(checks['checked_docs_sha256'], 'final checked documentation')
    if 'script_sha256' in checks:
        need(isinstance(checks.get('script_path'), str), 'Final QA script_path required with script_sha256')
        add(checks['script_path'], checks['script_sha256'], 'final QA generator')
    return {stage:{k:evaluations[stage]['metrics']['geography_blend'][k] for k in ('overall_mae', 'weighted_mae')} for stage in evaluations}


def source_closure(paths):
    found, todo = set(), list(paths)
    while todo:
        path = safe(ROOT, todo.pop())
        if path in found: continue
        need(path.is_file(), 'Missing source: '+str(path)); found.add(path)
        tree = ast.parse(path.read_text(encoding='utf-8-sig'))
        for node in ast.walk(tree):
            names = [a.name.split('.')[0] for a in node.names] if isinstance(node, ast.Import) else [node.module.split('.')[0]] if isinstance(node, ast.ImportFrom) and node.module else []
            for name in names:
                candidate = ROOT/'src'/(name+'.py')
                if candidate.is_file() and candidate not in found: todo.append(candidate)
    return found


def collect_files(expected):
    files, omitted = set(), []
    for folder, directories, names in os.walk(OUT, followlinks=False):
        parent = Path(folder); keep = []
        for name in directories:
            path = parent/name
            if name in OMIT_DIRS or name.startswith(('.', 'live_status')) or path.is_symlink(): omitted.append(rel(path))
            else: keep.append(name)
        directories[:] = keep
        for name in names:
            path = parent/name
            if path.is_symlink() or name.startswith(('.', 'live_status')) or path.suffix.lower() in {'.tmp', '.temp', '.lock', '.log', '.pyc', '.part'}: omitted.append(rel(path))
            else:
                need(path.suffix.lower() in SUFFIXES, 'Unclassified R6 artifact; specify copy policy first: '+rel(path))
                files.add(safe(ROOT, path))
    files.update(safe(ROOT, name) for name in REQUIRED_REPORTS)
    files.add(ROOT/'outputs/submission_v3.csv')
    files.update(p.resolve() for p in (ROOT/'reports').glob('*round6*') if p.is_file())
    sources = set((ROOT/'src').glob('*round6*.py'))|{Path(__file__).resolve()}
    for name in expected:
        path = ROOT/name
        if name.startswith('src/') and path.suffix == '.py': sources.add(path)
        elif name.startswith('reports/') and path.suffix.lower() in SUFFIXES: files.add(path.resolve())
        elif name.startswith(('outputs/geography/', 'data_external/geography/')) and path.suffix.lower() in {'.json', '.sha256', '.md', '.txt'}:
            files.add(path.resolve())  # named small evidence only; no geography tree scan
    files.update(source_closure(sources)); files.update(p.resolve() for p in ROOT.glob('requirements*.txt'))
    for name in ('reports/kaggle_submission_history.json', 'data_external/geography/ATTRIBUTION.md', 'solution_round6.ipynb'):
        if (ROOT/name).is_file(): files.add((ROOT/name).resolve())
    for path in files:
        name = rel(path)
        need(not name.startswith(('runs/', 'data_cache/', 'outputs/round2/', 'outputs/round3/baseline/')) and path.suffix.lower() not in {'.xlsx', '.pbf'}, 'Large/raw/prior dependency must remain external: '+name)
        need(not (name.startswith(('outputs/geography/', 'data_external/geography/')) and path.suffix.lower() in {'.pkl', '.jsonl', '.npy', '.cbm'}), 'Large geographic dependency must remain external: '+name)
    return sorted(files), sorted(omitted)


def record(path):
    path = safe(ROOT, path)
    return {'original_path':rel(path), 'original_absolute_path':str(path), 'sha256':sha(path), 'bytes':path.stat().st_size}


def build_archive_plan(registry_context=None):
    expected, reasons, add, mapping, contract = registry_collect()
    plan, frozen, accepted, release, evaluations = validate_contracts(add, mapping, contract)
    prior, sample = verify_prior(plan, add, contract)
    csv_proof = check_csv(release, sample)
    metrics = validate_reports(plan, accepted, release, evaluations, mapping, add)
    if registry_context is None:
        registry_path = ROOT/'runs/versions.json'
        registry_context = {'sha256':sha(registry_path), 'bytes':registry_path.stat().st_size, 'value':read(registry_path)}
    need(registry_context['sha256'] == release['versions_context_sha256'], 'Registry differs from release-time context')
    need(registry_context['value']['recommended_version'] == 'v2', 'Expected preserved v2 recommendation before v3 registration')
    entries = registry_context['value']['versions']
    need(len({x['version'] for x in entries}) == len(entries) and not any(x['version'] == 'v3' for x in entries), 'Prior registry contains duplicate/v3 entries')
    files, omitted = collect_files(expected)
    originals = {rel(path):record(path) for path in files}
    external = []
    for name, digest in sorted(expected.items()):
        if name in originals:
            need(originals[name]['sha256'] == digest, 'Source changed during collection: '+name); continue
        item = record(name); need(item['sha256'] == digest, 'Dependency changed during collection: '+name)
        backups = []
        for kind in ('v1', 'v2', 'round3_experiments', 'round4_experiments', 'round5_experiments'):
            path = safe(ROOT, Path('runs')/kind/name)
            if path.is_file() and sha(path) == digest: backups.append(rel(path))
        if name.startswith('runs/'): backups.append(name)
        item.update(reasons=sorted(reasons[name]), recorded_sha256=digest, existing_backups=sorted(set(backups)))
        external.append(item)
    return {'originals':originals, 'external_dependencies':external, 'omitted_paths':omitted,
      'prior_archives_verified':prior, 'registry_before':registry_context, 'registry_metrics':metrics,
      'release_path':'outputs/round6/release/release.json', 'release_sha256':sha(OUT/'release/release.json'),
      'plan_sha256':sha(OUT/'candidate_plan.json'), 'frozen_candidate_sha256':sha(OUT/'frozen_candidate.json'),
      'accepted_candidate_sha256':sha(OUT/'accepted_candidate.json'), 'submission':csv_proof,
      'historical_audit_policy':'Preserve draft proof unchanged; old prepare source was not preserved and is not rebound or certified for replay',
      'dependency_policy':'Only exact plan/freeze/acceptance/evaluation/prepared/release SHA contracts; raw/cache/large maps/old models external'}


def registry_entry(plan, manifest_sha):
    return {'version':'v3', 'status':'completed', 'label':'固定歷史 OSM 地理特徵融合；原 v2 趨勢一次',
      'submission':'runs/v3/outputs/submission_v3.csv', 'sha256':plan['submission']['sha256'], 'rows':47073,
      'snapshot':'runs/v3', 'snapshot_manifest_sha256':manifest_sha,
      'release_config':'runs/v3/outputs/round6/release/release.json', 'release_sha256':plan['release_sha256'],
      'experiment_round':6, 'candidate':'geography_blend', 'metrics':plan['registry_metrics'],
      'metric_protocol':'R6 same-stage annual rolling-origin MAE; not the older v2 short-window metric',
      'kaggle_score':None, 'kaggle_public_mae':None, 'kaggle_private_mae':None}


def expected_registry(plan, manifest_sha):
    before = plan['registry_before']['value']
    result = dict(before); result['versions'] = list(before['versions'])+[registry_entry(plan, manifest_sha)]
    result['recommended_version'] = 'v3'
    return result


def verify_registry(plan, manifest_sha):
    path = safe(ROOT, 'runs/versions.json'); current = read(path)
    if sha(path) == plan['registry_before']['sha256']:
        need(current == plan['registry_before']['value'], 'Prior registry content differs'); return 'before_registration'
    need(current == expected_registry(plan, manifest_sha), 'Registry changed outside this exact append-only v3 registration')
    return 'registered'


def verify_snapshot(folder):
    folder = safe(ROOT, folder); path = folder/MANIFEST
    need(folder.is_dir() and path.is_file(), 'Missing v3 snapshot')
    manifest = locked(path); plan = manifest['original_plan']
    need(manifest['version'] == 'v3' and manifest['is_release'] is True and manifest['accepted'] is True, 'Wrong snapshot kind')
    need(set(manifest['copied_files']) == set(plan['originals']), 'Copy inventory differs')
    actual = {p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}
    need(actual == set(plan['originals'])|{MANIFEST, RECEIPT, NOTICE_FILE, 'REGISTRY_BEFORE.json'}, 'Snapshot actual inventory differs')
    for name, item in manifest['copied_files'].items():
        original = plan['originals'][name]
        need(item['original_path'] == name and item['backup_path'] == 'runs/v3/'+name, 'Snapshot path record differs')
        need(item['original_sha256'] == item['backup_sha256'] == original['sha256'] and item['original_bytes'] == item['backup_bytes'] == original['bytes'], 'Original/copy proof differs')
        verify_file(folder, name, item['backup_sha256'], item['backup_bytes'])
    need(manifest['files_sha256'] == {name:item['backup_sha256'] for name,item in manifest['copied_files'].items()}, 'Compatibility hash inventory differs')
    need(manifest['external_dependencies'] == plan['external_dependencies'], 'Dependency inventory differs')
    verify_file(folder, 'REGISTRY_BEFORE.json', plan['registry_before']['sha256'], plan['registry_before']['bytes'])
    need(read(folder/'REGISTRY_BEFORE.json') == plan['registry_before']['value'], 'Registry snapshot differs')
    need((folder/NOTICE_FILE).read_bytes() == NOTICE.encode('utf-8'), 'Dependency notice differs')
    return manifest


def verify_current(plan):
    for name, item in plan['originals'].items(): verify_file(ROOT, name, item['sha256'], item['bytes'])
    for item in plan['external_dependencies']: verify_file(ROOT, item['original_path'], item['sha256'], item['bytes'])
    need(build_archive_plan(plan['registry_before']) == plan, 'Current contracts or exact artifact inventory changed')


def register(manifest):
    plan = manifest['original_plan']; msha = sha(DEST/MANIFEST)
    if verify_registry(plan, msha) == 'registered':
        print(json.dumps({'status':'already_registered', 'recommended_version':'v3'})); return
    # This is the only command which may change the registry. Snapshot and release
    # verification have already completed, and old entries are copied unchanged.
    path = safe(ROOT, 'runs/versions.json'); before = path.read_bytes()
    need(hashlib.sha256(before).hexdigest() == plan['registry_before']['sha256'], 'Registry changed before backup')
    backup = safe(ROOT, 'runs/versions.before_v3.'+plan['registry_before']['sha256']+'.json')
    if backup.exists(): need(backup.read_bytes() == before, 'Existing registry backup differs')
    else:
        with backup.open('xb') as handle: handle.write(before)
    temporary = safe(ROOT, 'runs/.versions-v3-staging-'+uuid.uuid4().hex+'.json')
    try:
        with temporary.open('xb') as handle: handle.write(encoded(expected_registry(plan, msha)))
        need(read(temporary) == expected_registry(plan, msha), 'Staged registry differs')
        need(sha(path) == plan['registry_before']['sha256'] and sha(DEST/MANIFEST) == msha, 'Registry/snapshot changed during registration')
        need(temporary.parent == path.parent and temporary.parent == (ROOT/'runs').resolve(), 'Unsafe registry update path')
        os.replace(temporary, path)
    except BaseException:
        print('Registration failed; backup and any staging retained: '+str(backup)+' ; '+str(temporary), flush=True); raise
    need(verify_registry(plan, msha) == 'registered', 'Registered state differs')
    print(json.dumps({'status':'registered', 'recommended_version':'v3', 'backup':rel(backup), 'registry_sha256':sha(path)}, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--verify', action='store_true', help='Read-only snapshot, originals, dependencies and release verification')
    group.add_argument('--register', action='store_true', help='Verify existing v3, back up registry, append v3 and recommend it')
    args = parser.parse_args(); destination = safe(ROOT, DEST); runs = safe(ROOT, 'runs')
    need(destination.parent == runs and destination.name == 'v3', 'Unsafe destination')
    if args.verify or args.register:
        manifest = verify_snapshot(destination); verify_current(manifest['original_plan'])
        state = verify_registry(manifest['original_plan'], sha(destination/MANIFEST))
        if args.register: register(manifest)
        else: print(json.dumps({'status':'verified', 'snapshot':'runs/v3', 'files':len(manifest['copied_files']), 'manifest_sha256':sha(destination/MANIFEST), 'registry_state':state}))
        return
    need(not destination.exists(), 'runs/v3 exists; use --verify. Never overwrite it')
    plan = build_archive_plan(); runs.mkdir(parents=True, exist_ok=True)
    copied_bytes = sum(item['bytes'] for item in plan['originals'].values())
    need(shutil.disk_usage(runs).free > copied_bytes+16*1024*1024, 'Insufficient snapshot disk space')
    staging = safe(runs, '.v3-staging-'+uuid.uuid4().hex); need(staging.parent == runs, 'Unsafe staging path'); staging.mkdir()
    try:
        copied = {}
        for name, item in sorted(plan['originals'].items()):
            source = safe(ROOT, name); target = safe(staging, name); target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target); verify_file(staging, name, item['sha256'], item['bytes'])
            copied[name] = {'original_path':name, 'original_absolute_path':str(source), 'original_sha256':item['sha256'], 'original_bytes':item['bytes'],
                            'backup_path':'runs/v3/'+name, 'backup_sha256':sha(target), 'backup_bytes':target.stat().st_size}
        before = (ROOT/'runs/versions.json').read_bytes()
        need(hashlib.sha256(before).hexdigest() == plan['registry_before']['sha256'], 'Registry changed while archiving')
        (staging/'REGISTRY_BEFORE.json').write_bytes(before)
        (staging/NOTICE_FILE).write_bytes(NOTICE.encode('utf-8'))
        manifest = {'format_version':1, 'version':'v3', 'is_release':True, 'accepted':True, 'experiment_round':6,
          'created_at_utc':datetime.now(timezone.utc).isoformat(), 'self_contained':False, 'dependency_notice':NOTICE_FILE,
          'copied_files':copied, 'files_sha256':{name:item['backup_sha256'] for name,item in copied.items()},
          'external_dependencies':plan['external_dependencies'], 'original_plan':plan, 'prior_archives_modified':False,
          'registry_modified_by_archive':False, 'uploaded_to_kaggle':False, 'manifest_self_excluded':True, 'manifest_sha256_receipt':RECEIPT}
        (staging/MANIFEST).write_bytes(encoded(manifest)); (staging/RECEIPT).write_text(sha(staging/MANIFEST)+'\n', encoding='ascii')
        verify_snapshot(staging); verify_current(plan)
        need(sha(ROOT/'runs/versions.json') == plan['registry_before']['sha256'], 'Registry changed before snapshot publication')
        need(not destination.exists() and staging.parent == destination.parent == runs, 'Destination appeared or unsafe publication path')
        staging.rename(destination)
    except BaseException:
        print('Archive not finalized; staging retained: '+str(staging), flush=True); raise
    verify_snapshot(destination)
    print(json.dumps({'status':'created', 'snapshot':'runs/v3', 'files':len(copied), 'copied_bytes':copied_bytes,
      'external_dependencies':len(plan['external_dependencies']), 'manifest_sha256':sha(destination/MANIFEST), 'registered':False}))


if __name__ == '__main__': main()
