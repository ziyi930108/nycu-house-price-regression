"""Byte-only snapshot for completed R25 pure-X research, never a price release.
Only --create/--verify publish or inspect a snapshot; --self-check is synthetic.
No array/model deserialization, training, prediction, price scoring or rule changes.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round25_research'
DEST = ROOT / 'runs/round25_research'
SOURCES = ('src/round25_address_features.py', 'src/round25_support_audit.py',
           'src/round25_address_geo.py', 'src/round25_experiment.py', 'src/archive_round25.py')
DOCS = ('reports/round25_research_plan.md', 'reports/round25_plan.md',
        'reports/round25_parser_review.md', 'reports/round25_address_support.md',
        'reports/round25_work_log.md', 'reports/round25_report.md',
        'reports/round25_results.json', 'reports/round25_results.sha256')
SUPPORT = 'outputs/round25_research/address_support.json'
EDGE_CASES = 'outputs/round25_research/parser_edge_cases.json'
RESULTS = 'reports/round25_results.json'
REPORT = 'reports/round25_report.md'
WORK_LOG = 'reports/round25_work_log.md'
RESEARCH_FILES = (SUPPORT, SUPPORT[:-5]+'.sha256',
                  'outputs/round25_research/address_support.log',
                  EDGE_CASES, EDGE_CASES[:-5]+'.sha256')
REASON = 'sparse_usable_address_support_and_semantic_limits'
FIXED_REPORT_SHA = '867238857c051f29f78bc6a9f6dbf918ccd4f6179269f0d684a0753cc3f2a7ef'
FIXED_RESULTS_SHA = '247daa1254d42e20f5001161e31c0e47a2a002245d4115096bf244d36166748d'
FIXED_SUPPORT_SHA = 'b47dadbbfb2445b5da930ff9dfe7643577b1467b1689550eb3452426a46c78a5'
V3_SNAPSHOT = 'runs/v3/snapshot_manifest.json'
V3_SNAPSHOT_SHA = '77420a2b71cdb64886998e8f621065735b08ba57ec9d3fce5201edee6050a525'
PROTECTED = ('outputs/submission_v2.csv', 'outputs/submission_v3.csv')
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = """# R25 research-only snapshot

The fixed three-feature address pack was not promoted to a price experiment.
Coverage/support are descriptive pure-X evidence, not price acceptance gates or
proof that the features have no predictive value. Model helper/driver and plan
Markdown are preserved as unexecuted drafts; there is no formal price-plan JSON,
trained model, price evaluation, prediction CSV, upload or release for R25.
The existing v3 recommendation is unchanged.

This snapshot copies all five owned sources, every R25 research output and report,
including artificial parser observations, the single successful support audit,
receipts/logs and the CLOSED work log. Large parsed inputs and saved indices,
legacy sources and immutable authority manifests are external SHA/size bindings.
The original canonical deduplication historically used targets; this research
only reused saved indices and never recomputed that deduplication or read labels.

Only JSON and raw bytes are read by this archiver. No pickle, array or model is
loaded and no new support statistic, MAE or gate is computed. CLOSED must occur
exactly once in the same work-log bytes copied. No file is overwritten/deleted;
partial staging is preserved and causes refusal. Registry is creation-time
context only, and CURRENT_STATUS is excluded from permanent verification.
This is an auditable research snapshot, not a self-contained model release.
"""

def need(ok, message):
    if not ok:
        raise RuntimeError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2,
                       allow_nan=False) + '\n').encode('utf-8')

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def normalized(name):
    need(isinstance(name, str), 'Path must be a relative string')
    name = name.replace('\\', '/')
    need(name and name == name.strip() and ':' not in name and '\x00' not in name,
         'Unsafe relative path: ' + repr(name))
    need(not PurePosixPath(name).is_absolute()
         and all(part not in ('', '.', '..') for part in name.split('/')),
         'Unsafe relative path: ' + name)
    return name

def safe(name, base=ROOT, exists=True):
    name = normalized(name)
    root = Path(base).resolve()
    path = (root / name).resolve()
    need(path.is_relative_to(root), 'Path escapes declared root: ' + name)
    if exists:
        need(path.is_file(), 'Missing file: ' + str(path))
    return path

def rel(path):
    return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()

def locked(path):
    path = Path(path)
    receipt = path.with_suffix('.sha256')
    need(path.is_file() and receipt.is_file(), 'Missing JSON/receipt: ' + str(path))
    need(receipt.read_text(encoding='ascii').strip() == sha(path), 'Receipt differs: ' + str(path))
    return read(path)

class Bindings:
    """Merge only authoritative current maps, with mutable exceptions explicit."""
    def __init__(self):
        self.values, self.case_names, self.mutable_declarations = {}, {}, {}

    def declare(self, name, digest):
        name = normalized(name)
        need(isinstance(digest, str) and re.fullmatch('[0-9a-f]{64}', digest), 'Invalid SHA: ' + name)
        previous = self.case_names.get(name.casefold())
        need(previous is None or previous == name, 'Case-ambiguous dependency: ' + name)
        self.case_names[name.casefold()] = name
        if name.casefold() in {p.casefold() for p in MUTABLE}:
            need(name in MUTABLE, 'Case-ambiguous mutable path')
            self.mutable_declarations.setdefault(name, set()).add(digest)
            return False
        need(name not in self.values or self.values[name]['sha256'] == digest,
             'Conflicting authority SHA: ' + name)
        return True

    def add(self, name, expected=None):
        name = normalized(name)
        if expected is not None and not self.declare(name, expected):
            return None  # Historical context, deliberately not a live guard.
        if name in self.values:
            return safe(name)
        path = safe(name)
        digest = sha(path)
        if not self.declare(name, digest):
            return None
        need(expected is None or expected == digest, 'Changed dependency: ' + name)
        self.values[name] = {'sha256': digest, 'bytes': path.stat().st_size}
        return path

    def mapping(self, values):
        need(isinstance(values, dict), 'Expected path-to-SHA map')
        for name, digest in values.items():
            self.add(name, digest)

    def authority(self, name):
        path = self.add(name)
        need(path is not None, 'Mutable file cannot be authority')
        value = locked(path)
        self.add(rel(path.with_suffix('.sha256')))
        return value

def validate_current_paths(names):
    for name in names:
        normalized(name)
        need(not any(part.startswith(('.', '_stage_')) for part in name.split('/'))
             and not name.lower().endswith(('.tmp','.part','.lock')),
             'Current incomplete staging must be preserved and inspected: '+name)

def closed_work_log_sha256(raw):
    """Require one exact CLOSED line in the same bytes bound for later copying."""
    need(isinstance(raw, bytes), 'Work log validation requires immutable bytes')
    text = raw.decode('utf-8-sig')
    marker_lines = [line for line in text.splitlines() if 'work_log_status' in line.casefold()]
    need(marker_lines == ['<!-- WORK_LOG_STATUS: CLOSED -->'],
         'Work log must contain exactly one exact CLOSED marker and no other status marker')
    return hashlib.sha256(raw).hexdigest()

def copy_scope(name):
    name = normalized(name)
    return (name in SOURCES or name.startswith('outputs/round25_research/')
            or name.startswith('reports/round25_'))

def inventory():
    need(OUT.is_dir(), 'Missing R25 research')
    need(not (ROOT/'outputs/round25').exists(), 'Unexpected formal R25 experiment directory')
    need(not list((ROOT/'reports').glob('.round25-report-*')), 'Partial report staging; preserve it')
    discovered = {rel(p) for p in (ROOT/'src').glob('round25_*.py')} | {'src/archive_round25.py'}
    need(discovered == set(SOURCES), 'Unexpected or missing owned R25 source')
    paths = set(OUT.rglob('*')) | set((ROOT/'reports').glob('round25_*'))
    paths.update(ROOT/name for name in SOURCES)
    validate_current_paths({p.relative_to(OUT).as_posix() for p in OUT.rglob('*')})
    files = set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink not an archive artifact')
        need(not path.is_dir() or path.is_relative_to(OUT), 'Unexpected report directory')
        if path.is_file():
            name = rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R25 copy scope')
            need(path.suffix.lower() not in ('.cbm','.npy','.npz','.pkl','.csv','.tmp','.part','.lock')
                 and path.name != 'model.txt', 'Unexpected model/array/submission/partial artifact: '+name)
            files.add(name)
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= files, 'Required research/document missing')
    return files


def bind_record(bindings, value):
    path = bindings.add(value['path'], value['sha256'])
    need(path is not None, 'Research input cannot be mutable context')
    if 'bytes' in value:
        need(type(value['bytes']) is int and path.stat().st_size == value['bytes'], 'Declared input bytes differ')


def validate_support(value):
    need(value['round'] == 25 and value['status'] == 'pure_X_support_inventory_complete', 'Wrong support status')
    need(value['all_read_sources_and_inputs_unchanged'] is True
         and value['input_before'] == value['input_after'], 'Audit inputs changed during execution')
    scope = value['read_scope']
    need(scope['parsed_splits_loaded'] == ['train','valid','test']
         and all(scope[k] is False for k in ('original87_features_loaded','price_cache_or_excel_loaded',
             'model_or_prediction_loaded','labels_used','model_training_or_price_scoring',
             'canonical_dedup_recomputed','external_answers_used'))
         and scope['existing_target_containing_dedup_index_reused'] is True, 'Wrong audit read boundary')
    need(set(value['inputs']) == {'canonical_keep','fit','query','train_parsed','valid_parsed','test_parsed'},
         'Wrong audit input inventory')
    need(set(value['source_sha256']) == {'src/round25_support_audit.py','src/round25_address_features.py',
         'src/geo_address_audit.py','src/house_features.py'}, 'Wrong research source closure')
    for split in ('train','valid','test'):
        item = value['reconstruction_checks'][split]
        need(item['feature_values_dtypes_index_exact'] is True and item['query_input_unchanged'] is True
             and item['original_87_cache_loaded'] is False, 'Saved reconstruction check did not pass')


def validate_report(report, audit):
    need(report['round'] == 25 and report['status'] == 'research_only_not_promoted'
         and report['reason'] == REASON and report['recommended_submission'] == 'outputs/submission_v3.csv',
         'Wrong research conclusion')
    for key in ('promoted_to_price_experiment','new_submission_generated','kaggle_uploaded',
                'formal_driver_research_dependencies_ready'):
        need(report[key] is False, 'Report invents promotion/release: '+key)
    need(report['private_mae'] is None and report['model_drafts_source_review_only'] is True
         and report['support_counts_are_descriptive_not_price_gates'] is True
         and report['no_price_effect_claim'] is True, 'Wrong result interpretation')
    need(set(report['price_stages']) == {'development','confirmation','auxiliary','final'}, 'Wrong stage inventory')
    for stage in report['price_stages'].values():
        need(stage['executed'] is False and stage['mae'] is None, 'Price stage must remain unexecuted/null')
    need(report['report_sha256'] == FIXED_REPORT_SHA and report['audit_result_sha256'] == FIXED_SUPPORT_SHA,
         'Wrong saved report/audit identity')
    p = audit['populations']['development_query']
    s = audit['support']['development_fit_to_query']
    summary = report['development_query_summary']
    expected = {'rows':p['rows'], 'eligible':p['eligible']['rows'],
        'path_seen':s['keys']['addr_lane_path']['seen']['rows'],
        'path_support_ge10':s['keys']['addr_lane_path']['supported_by_at_least_10_fit_rows']['rows'],
        'block_seen':s['keys']['addr_house_block100']['seen']['rows'],
        'block_support_ge10':s['keys']['addr_house_block100']['supported_by_at_least_10_fit_rows']['rows'],
        'block_distinct_main_ge2':s['block_with_at_least_2_distinct_fit_main_numbers']['rows'],
        'block_support_ge10_and_distinct_main_ge2':s['block_with_both_10_fit_rows_and_2_distinct_main_numbers']['rows']}
    need(all(summary[k] == v for k,v in expected.items()), 'Report changed saved descriptive counts')
    need(summary['block_support_ge10_all_query_fraction'] == expected['block_support_ge10']/expected['rows'],
         'Wrong descriptive denominator')


def context():
    # CURRENT_STATUS is deliberately not read; registry can change after creation.
    name = 'runs/versions.json'
    return {name:{'sha256':sha(safe(name)), 'bytes':safe(name).stat().st_size}}


def released_csv_records(snapshot):
    need(snapshot['version'] == 'v3' and snapshot['is_release'] is True, 'Wrong released snapshot')
    v3 = snapshot['copied_files']['outputs/submission_v3.csv']
    need(v3['original_path'] == 'outputs/submission_v3.csv'
         and v3['original_sha256'] == v3['backup_sha256']
         and v3['original_bytes'] == v3['backup_bytes'], 'Released v3 CSV identity differs')
    v2 = [item for item in snapshot['external_dependencies']
          if item['original_path'] == 'outputs/submission_v2.csv']
    need(len(v2) == 1 and v2[0]['recorded_sha256'] == v2[0]['sha256'], 'Released v2 CSV identity differs')
    v2 = v2[0]
    records = [{'path':v3['original_path'], 'sha256':v3['original_sha256'], 'bytes':v3['original_bytes']},
               {'path':v3['backup_path'], 'sha256':v3['backup_sha256'], 'bytes':v3['backup_bytes']},
               {'path':v2['original_path'], 'sha256':v2['sha256'], 'bytes':v2['bytes']}]
    records.extend({'path':name, 'sha256':v2['sha256'], 'bytes':v2['bytes']} for name in v2['existing_backups'])
    return records


def collect():
    before = context(); files = inventory(); bindings = Bindings()
    need(read(safe('runs/versions.json'))['recommended_version'] == 'v3', 'Registry must still recommend v3 at creation')
    audit = bindings.authority(SUPPORT)
    need(sha(safe(SUPPORT)) == FIXED_SUPPORT_SHA, 'Fixed support artifact changed')
    validate_support(audit)
    for key in ('source_sha256','evidence_sha256'):
        bindings.mapping(audit[key])
    for item in audit['inputs'].values():
        bind_record(bindings, item)
    declared = set(audit['source_sha256']) | set(audit['evidence_sha256']) | {r['path'] for r in audit['inputs'].values()}
    need(set(audit['input_before']) == declared, 'Unexplained or omitted audit read dependency')
    for name, item in audit['input_before'].items():
        need(item['path'] == name, 'Audit input-map identity mismatch')
        bind_record(bindings, item)
    edge = bindings.authority(EDGE_CASES)
    need(edge['round'] == 25 and edge['status'] == 'historical_artificial_parser_observations_recorded', 'Wrong parser evidence')
    source = edge['parser_source']
    need(source['source_unchanged'] is True and source['sha256_before_execution'] == source['sha256_after_execution'],
         'Parser changed during artificial observations')
    bindings.add(source['path'], source['sha256_after_execution'])
    need(all(edge['execution'][k] is False for k in ('real_inputs_read','models_loaded','training_or_scoring_performed',
         'parser_rerun_for_this_record')), 'Parser record is not the declared historical artificial observation')
    report = bindings.authority(RESULTS)
    need(sha(safe(RESULTS)) == FIXED_RESULTS_SHA, 'Final report JSON changed')
    validate_report(report, audit)
    bindings.mapping(report['source_input_sha256'])
    bindings.add(REPORT, report['report_sha256'])
    bindings.add(V3_SNAPSHOT, V3_SNAPSHOT_SHA)
    snapshot = bindings.authority(V3_SNAPSHOT)
    for record in released_csv_records(snapshot):
        bind_record(bindings, record)
    lograw = safe(WORK_LOG).read_bytes()
    bindings.add(WORK_LOG, closed_work_log_sha256(lograw))
    for name in sorted(files):
        bindings.add(name)
    need(not any(copy_scope(n) and n not in files for n in bindings.values), 'Owned dependency omitted from inventory')
    copied = {n:bindings.values[n] for n in sorted(files)}
    external = {n:v for n,v in sorted(bindings.values.items()) if n not in files}
    need(context() == before, 'Registry context changed during collection')
    state = {'round':25, 'is_release':False, 'research_status':'research_only_not_promoted', 'reason':REASON,
        'promoted_to_price_experiment':False, 'price_training_performed':False, 'price_evaluation_performed':False,
        'new_submission_generated':False, 'kaggle_uploaded':False, 'recommended_version':'v3',
        'model_drafts_unexecuted':True, 'arrays_or_models_deserialized':False,
        'support_sha256':copied[SUPPORT]['sha256'], 'parser_evidence_sha256':copied[EDGE_CASES]['sha256'],
        'report_json_sha256':copied[RESULTS]['sha256'], 'report_markdown_sha256':copied[REPORT]['sha256'],
        'source_sha256':copied['src/archive_round25.py']['sha256'], 'closed_work_log_sha256':copied[WORK_LOG]['sha256'],
        'protected_artifacts':{n:external[n] for n in PROTECTED}, 'mutable_context_at_creation':before,
        'mutable_declarations_not_live_guards':{n:sorted(v) for n,v in bindings.mutable_declarations.items()}}
    return copied, external, state


def manifest_structure(manifest):
    need(manifest['kind'] == 'round25_research_not_promoted', 'Wrong snapshot kind')
    state = manifest['state']
    need(state['round'] == 25 and state['is_release'] is False
         and state['research_status'] == 'research_only_not_promoted' and state['reason'] == REASON
         and state['recommended_version'] == 'v3' and state['model_drafts_unexecuted'] is True
         and all(state[k] is False for k in ('promoted_to_price_experiment','price_training_performed',
             'price_evaluation_performed','new_submission_generated','kaggle_uploaded','arrays_or_models_deserialized')),
         'Snapshot invents a price experiment')
    copied, external = manifest['copied_files'], manifest['external_dependencies']
    need(set(copied).isdisjoint(external), 'Copy/external overlap')
    names = set(copied)|set(external)
    need(len({normalized(n).casefold() for n in names}) == len(names), 'Case-ambiguous paths')
    need(not {n.casefold() for n in names}&{n.casefold() for n in MUTABLE}, 'Mutable permanent guard')
    need(set(SOURCES).union(DOCS, RESEARCH_FILES) <= set(copied) and all(copy_scope(n) for n in copied), 'Wrong inventory')
    need(all(not copy_scope(n) for n in external), 'Owned artifact left external')
    need(manifest['copied_file_count'] == len(copied) and manifest['external_dependency_count'] == len(external)
         and manifest['copied_bytes'] == sum(i['bytes'] for i in copied.values()), 'Wrong manifest counts')
    for item in {**copied,**external}.values():
        need(type(item['bytes']) is int and item['bytes'] >= 0 and re.fullmatch('[0-9a-f]{64}',item['sha256']), 'Invalid identity')
    for key,name in [('support_sha256',SUPPORT),('parser_evidence_sha256',EDGE_CASES),
                     ('report_json_sha256',RESULTS),('report_markdown_sha256',REPORT),
                     ('source_sha256','src/archive_round25.py'),('closed_work_log_sha256',WORK_LOG)]:
        need(state[key] == copied[name]['sha256'], 'Authority mismatch: '+key)
    need(state['protected_artifacts'] == {n:external[n] for n in PROTECTED}, 'Protected authority mismatch')
    return copied, external


def verify_at(folder, originals=True):
    """Verify a snapshot using only JSON and raw bytes, in a fresh process."""
    folder = Path(folder).resolve()
    need(folder.is_relative_to((ROOT / 'runs').resolve()), 'Snapshot outside workspace runs')
    manifest = locked(folder / 'snapshot_manifest.json')
    copied, external = manifest_structure(manifest)
    expected = {'snapshot_manifest.json', 'snapshot_manifest.sha256', 'SNAPSHOT_README.md'}
    for name, item in copied.items():
        need(item['backup_path'] == rel(DEST / 'files' / name), 'Unexpected final backup path')
        backup = safe('files/' + normalized(name), folder)
        need(backup.stat().st_size == item['bytes'] and sha(backup) == item['sha256'], 'Snapshot copy changed: ' + name)
        if originals:
            original = safe(name)
            need(original.stat().st_size == item['bytes'] and sha(original) == item['sha256'], 'Frozen original changed: ' + name)
        expected.add('files/' + name)
    for name, item in external.items():
        path = safe(name)
        need(path.stat().st_size == item['bytes'] and sha(path) == item['sha256'], 'External dependency changed: ' + name)
    notice = folder / 'SNAPSHOT_README.md'
    need(notice.read_bytes() == NOTICE.encode('utf-8') and sha(notice) == manifest['notice_sha256'], 'Snapshot notice differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()} == expected,
         'Snapshot contains missing/extra files')
    if originals:
        need(inventory() == set(copied), 'Original R25 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}

def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round25_research_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round25_research_staging_' + uuid.uuid4().hex)
    need(staging.resolve().is_relative_to((ROOT / 'runs').resolve()) and not staging.exists(), 'Unsafe staging')
    staging.mkdir(parents=True)
    records = {}
    for name, item in copied.items():
        source, target = safe(name), safe('files/' + name, staging, exists=False)
        need(not target.exists(), 'Copy destination exists')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        need(source.stat().st_size == target.stat().st_size == item['bytes']
             and sha(source) == sha(target) == item['sha256'], 'Source/copy changed: ' + name)
        records[name] = {**item, 'original_sha256': item['sha256'], 'backup_path': rel(DEST / 'files' / name)}
    notice = staging / 'SNAPSHOT_README.md'
    notice.write_bytes(NOTICE.encode('utf-8'))
    manifest = {'kind': 'round25_research_not_promoted',
                'created_at_utc': datetime.now(timezone.utc).isoformat(), 'state': state,
                'copied_files': records, 'external_dependencies': external,
                'copied_file_count': len(records), 'external_dependency_count': len(external),
                'copied_bytes': sum(item['bytes'] for item in records.values()),
                'notice_sha256': sha(notice), 'not_self_contained': True,
                'failed_staging_preserved': True, 'registry_and_mutable_status_not_modified_or_bound': True}
    path = staging / 'snapshot_manifest.json'
    with path.open('xb') as handle:
        handle.write(encoded(manifest))
    with path.with_suffix('.sha256').open('x', encoding='ascii') as handle:
        handle.write(sha(path) + '\n')
    # Verify all bytes/authorities and exact inventory before the final rename.
    verify_at(staging)
    need(not DEST.exists(), 'Concurrent destination appeared; staging preserved')
    need(context() == state['mutable_context_at_creation'], 'Mutable context changed during archive copy')
    staging.rename(DEST)
    print(json.dumps({'status': 'created', 'snapshot_sha256': sha(DEST / 'snapshot_manifest.json'),
                      'copied_files': len(records), 'external_dependencies': len(external),
                      'copied_bytes': manifest['copied_bytes'], 'research_not_promoted': True,
                      'recommended_version': 'v3'}, ensure_ascii=False), flush=True)

def self_check():
    checks = []
    marker = b'<!-- WORK_LOG_STATUS: CLOSED -->\n'
    need(closed_work_log_sha256(marker) == hashlib.sha256(marker).hexdigest(), 'Same-byte CLOSED check')
    checks.append('exact_closed_same_bytes')
    for label, function in [('open',lambda:closed_work_log_sha256(b'<!-- WORK_LOG_STATUS: OPEN -->')),
                            ('duplicate',lambda:closed_work_log_sha256(marker+marker)),
                            ('parent_path',lambda:normalized('../old')),
                            ('absolute_path',lambda:normalized('C:/old'))]:
        try: function()
        except RuntimeError: checks.append(label+'_rejected')
        else: raise AssertionError(label)
    need(copy_scope('src/archive_round25.py') and copy_scope('outputs/round25_research/address_support.json')
         and not copy_scope('src/archive_round23.py') and not copy_scope('CURRENT_STATUS.md'), 'Copy boundary')
    checks.append('bounded_copy_scope')
    b = Bindings(); b.declare('src/example.py','a'*64)
    b.values['src/example.py'] = {'sha256':'a'*64,'bytes':1}
    try: b.declare('src/example.py','b'*64)
    except RuntimeError: checks.append('conflicting_hash_rejected')
    else: raise AssertionError('Conflicting hash accepted')
    need(b.declare('runs/versions.json','a'*64) is False and not b.values.get('runs/versions.json'), 'Registry is context only')
    checks.append('registry_not_permanent_dependency')
    return {'status':'passed','check_count':len(checks),'checks':checks,
            'actual_archive_created_or_verified':False,'data_or_models_loaded':False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create',action='store_true'); mode.add_argument('--verify',action='store_true')
    mode.add_argument('--self-check',action='store_true'); args = parser.parse_args()
    if args.self_check: print(json.dumps(self_check(),ensure_ascii=False))
    elif args.verify: print(json.dumps(verify_at(DEST),ensure_ascii=False))
    else: create()
