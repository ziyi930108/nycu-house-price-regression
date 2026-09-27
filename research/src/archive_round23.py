"""Freeze completed R23 research without inventing a house-price experiment.
Explicit --create / --verify / --self-check. Standard-library JSON and byte
hashes only; never deserialize a CBM/array or fit, predict, score or publish.
"""
from __future__ import annotations
import argparse, ast, copy, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/round23_research'
DEST = ROOT / 'runs/round23_research'
SOURCES = ('src/round23_ctr_feasibility.py', 'src/archive_round23.py')
DOCS = ('reports/round23_work_log.md', 'reports/round23_report.md',
        'reports/round23_results.json', 'reports/round23_results.sha256',
        'reports/round23_research_review.md')
FEAS = 'outputs/round23_research/ctr_feasibility.json'
BASELINE = 'outputs/round23_research/baseline_categorical_config.json'
HISTORY = 'outputs/round23_research/categorical_history.json'
REVIEW = 'outputs/round23_research/research_review.json'
QA_PATH = 'outputs/round23_research/report_artifact_qa.json'
REASON = 'requested_ctr_resolution_not_verified_in_native_descriptors'
FIXED_RECIPE_SHA = 'cbc07a47e32cbb07fc60b3a433bed355e2cf7cf9770d315b6561cee5018f8502'
FIXED_FEAS_SHA = '4249b1ecb056553b73dbcd9bbbde802794b8b21db565b25502f6d0b8a7a3aac9'
RESEARCH_FILES = tuple(p for n in (FEAS, BASELINE, HISTORY, REVIEW, QA_PATH)
                       for p in (n, n[:-5]+'.sha256')) + (
    'outputs/round23_research/ctr_feasibility.log',
    'outputs/round23_research/mae_control.cbm',
    'outputs/round23_research/ctr8_candidate.cbm')
FEEDBACK = ('outputs/kaggle_feedback/v3_public_20260925.json',
            'outputs/kaggle_feedback/leaderboard_20260925.json')
MUTABLE = {'runs/versions.json', 'CURRENT_STATUS.md'}
NOTICE = """# R23 research-only snapshot

The fixed artificial probe is not promoted to a house-price candidate.
Both two-tree artificial models were fitted, saved and loaded. The original
counter records one fully verified model (control); candidate status remains
started. Its global target-border count is eight while both target-based
simple/combinations native descriptors still record one. This is not evidence
that GPU fitting is unsupported, nor that every underlying CTR calculation
is unchanged. No weaker assertion, retry, alternate value or border search ran.

No formal house-price plan, fit, evaluation, acceptance gate, new prediction
CSV or release exists for R23. The recommendation remains the existing v3.
Historical Public feedback is context, not an R23 price score; Private is unknown.

This archive copies both owned sources, the complete R23 research directory
(including artificial CBMs, partial failure JSON/log, receipts, independent
review and report QA), and R23 reports. All declared history/source/config,
local package source, original baseline CBM, its explicit snapshot backup,
snapshot receipt, protected v2/v3 CSVs and feedback evidence are external
SHA/size dependencies. The entire old v3 release tree is not duplicated.
This is not a self-contained model release.

Only JSON and bytes are read. No model or array is deserialized and no price
error is computed. CLOSED must occur exactly once in the same work-log bytes
copied. Missing artifacts and temporary staging cause a hard refusal; files
are never overwritten or deleted. versions.json and CURRENT_STATUS.md are
creation-time context only, never permanent verification dependencies.
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
    return (name in SOURCES or name.startswith('outputs/round23_research/')
            or name.startswith('reports/round23_'))


def inventory():
    need(OUT.is_dir(), 'Missing R23 research')
    need(not (ROOT/'outputs/round23').exists(), 'Unexpected formal R23 experiment directory')
    need(not list((ROOT/'reports').glob('.round23-report-*')), 'Partial report staging; preserve it')
    paths = set(OUT.rglob('*')) | set((ROOT/'reports').glob('round23_*'))
    paths.update(ROOT/name for name in SOURCES)
    validate_current_paths({p.relative_to(OUT).as_posix() for p in OUT.rglob('*')})
    files=set()
    for path in paths:
        need(not path.is_symlink(), 'Symlink not an archive artifact')
        need(not path.is_dir() or path.is_relative_to(OUT), 'Unexpected report directory')
        if path.is_file():
            name=rel(path)
            need(copy_scope(name) and name not in MUTABLE, 'Artifact outside R23 copy scope')
            need(path.suffix.lower() not in ('.npy','.npz','.pkl','.csv','.tmp','.part','.lock'),
                 'Unexpected array/submission/partial artifact: '+name)
            files.add(name)
    need(set(SOURCES).union(DOCS,RESEARCH_FILES)<=files, 'Required R23 research/document missing')
    need({n for n in files if n.endswith('.cbm')}=={
        'outputs/round23_research/mae_control.cbm','outputs/round23_research/ctr8_candidate.cbm'},
        'Only the two artificial CBMs belong to R23')
    return files


def ctr_evidence(native):
    """Reparse saved JSON descriptors, not a CBM or the original recipe."""
    count=native['ctr_target_border_count']
    need(type(count) is int and count>0, 'Invalid global count')
    value={'global_ctr_target_border_count':count,'target_count_values':{}}
    for family in ('simple_ctr','combinations_ctr'):
        rows=[]
        need(isinstance(native[family],list) and native[family], 'Missing native CTR family')
        for descriptor in native[family]:
            need(isinstance(descriptor,str), 'CTR descriptor not a string')
            tokens=descriptor.split(':');counts=[s.split('=',1)[1] for s in tokens if s.startswith('TargetBorderCount=')]
            need(len(counts)<=1, 'Duplicate target-border field')
            number=int(counts[0]) if counts else None
            need(number is None or number>0, 'Invalid descriptor count')
            rows.append({'descriptor':descriptor,'ctr_type':tokens[0],'target_border_count':number})
        value[family]=rows
        value['target_count_values'][family]=sorted({r['target_border_count'] for r in rows if r['target_border_count'] is not None})
    return value


def validate_feasibility(feas):
    need(feas['round']==23 and feas['status']=='synthetic_GPU_compatibility_failed', 'Wrong research outcome')
    need(feas['synthetic_only'] is True and feas['property_X_y_models_or_predictions_loaded'] is False
         and feas['property_MAE_computed'] is False and feas['query_labels_used_for_training'] is False,
         'Real-property access/scoring is not part of this recipe')
    need(feas['planned_synthetic_model_count']==2 and feas['synthetic_model_count']==1
         and feas['fit_rows']==512 and feas['query_rows']==32 and feas['feature_count']==95
         and feas['categorical_feature_count']==23 and feas['synthetic_seed']==20260925, 'Artificial scope changed')
    need(feas['candidate_passed_parameter_changes']==['ctr_target_border_count']
         and feas['declared_candidate_ctr_target_border_count']==8
         and feas['feasibility_overrides']=={'iterations':2,'verbose':False}
         and feas['border_values_searched'] is False and feas['alternate_values_attempted'] is False
         and feas['no_alternative_value_or_retry_attempted'] is True, 'Search/retry/recipe changed')
    base=feas['baseline_parameters'];candidate=feas['candidate_parameters']
    need(candidate==dict(base,ctr_target_border_count=8) and 'ctr_target_border_count' not in base
         and base['loss_function']==base['eval_metric']=='MAE' and base['task_type']=='GPU'
         and base['iterations']==2999 and base['learning_rate']==.065 and base['depth']==8
         and base['boosting_type']=='Plain' and base['thread_count']==4, 'Wrong baseline or extra changes')
    need(feas['failure_phase']=='verify_artificial_model:ctr8_candidate'
         and feas['error']['type']=='ValueError'
         and feas['error']['message']=='Target-based native simple_ctr does not reflect the global target-border count',
         'This archive is only for the observed descriptor mismatch')
    need(set(feas['models'])=={'mae_control','ctr8_candidate'}, 'Artificial model inventory changed')
    for name,expected in [('mae_control',1),('ctr8_candidate',8)]:
        record=feas['models'][name]
        need(record['fit_rows']==512 and record['query_rows']==32, 'Artificial partition changed')
        params=base if name=='mae_control' else candidate
        need(record['passed_parameters']==dict(params,iterations=2,verbose=False), 'Artificial passed params changed')
        need(record['model_path']=='outputs/round23_research/'+name+'.cbm'
             and re.fullmatch('[0-9a-f]{64}',record['model_sha256']), 'Artificial model identity missing')
        for side in ('fitted','saved'):
            native=record[side+'_native_parameters'];parsed=ctr_evidence(native)
            need(parsed==record[side+'_ctr_evidence'], 'Captured CTR parsing differs')
            need(parsed['global_ctr_target_border_count']==expected
                 and parsed['target_count_values']=={'simple_ctr':[1],'combinations_ctr':[1]}, 'Native mismatch evidence changed')
            need(native['iterations']==2 and native['depth']==8 and native['task_type']=='GPU'
                 and native['loss_function']==native['eval_metric']=='MAE', 'Wrong synthetic native model')
            need(all(r['target_border_count'] is None for family in ('simple_ctr','combinations_ctr')
                     for r in parsed[family] if r['ctr_type']=='FeatureFreq'), 'Absent field cannot be zero')
    control,candidate=feas['models']['mae_control'],feas['models']['ctr8_candidate']
    need(control['status']=='passed' and control['saved_reload_predictions_exact'] is True
         and control['tree_count']==2 and control['feature_count']==95
         and len(control['categorical_positions'])==23, 'Control verification incomplete')
    need(candidate['status']=='started' and 'saved_reload_predictions_exact' not in candidate
         and 'synthetic_query_prediction_sha256' not in candidate, 'Do not invent candidate success fields')


def validate_review(review,feas):
    need(review['round']==23 and review['status']=='review_completed'
         and review['research_decision']=='not_advanced_to_price_candidate' and review['reason']==REASON,
         'Independent review decision differs')
    need(review['artificial_fits_completed']==review['artificial_models_saved_and_loaded']==2
         and review['fully_verified_models']==1, 'Review counts differ')
    for name,record in feas['models'].items():
        need(review['native_ctr_evidence'][name]=={k:record[k] for k in ('fitted_ctr_evidence','saved_ctr_evidence')},
             'Review native evidence differs')
        need(review['artificial_model_artifacts_from_original_evidence'][name]=={
            'model_path':record['model_path'],'model_sha256':record['model_sha256'],'status':record['status']},
             'Review artificial identity differs')
    need(review['error']==feas['error'] and review['failure_phase']==feas['failure_phase'], 'Review failure differs')
    interp=review['interpretation']
    need(interp['parameter_accepted_and_artificial_fit_completed'] is True
         and all(interp[k] is False for k in ('gpu_parameter_unsupported_proven','all_underlying_ctr_computation_unchanged_proven',
             'intended_eight_target_borders_verified_in_native_descriptors','retry_attempted','alternate_value_attempted',
             'criterion_weakened','parser_misread_identified')), 'Overstated review conclusion')
    need(review['source_sha256']==feas['source_sha256'] and review['input_sha256']==feas['input_sha256'], 'Review source differs')


def bind_record(bindings,value):
    path=bindings.add(value['path'],value['sha256'])
    if path is None:
        need(value['path'] in MUTABLE, 'Unexpected unbound artifact')
        current=safe(value['path'])
        need(sha(current)==value['sha256'], 'Mutable context changed since QA')
        if 'bytes' in value:need(current.stat().st_size==value['bytes'], 'Mutable context bytes changed')
    elif 'bytes' in value:need(path.stat().st_size==value['bytes'], 'Declared bytes differ: '+value['path'])
    return path


def bind_maps(bindings,value):
    for key in ('source_sha256','input_sha256','evidence_sha256'):
        if key in value:bindings.mapping(value[key])
    for item in value.get('inputs',{}).values():bind_record(bindings,item)


def validate_baseline(base,feas):
    need(base['round']==23 and base['status']=='baseline_native_settings_read_only', 'Wrong baseline probe')
    need(base['configuration']['effective_R6_passed_parameters']==feas['baseline_parameters'], 'Baseline parameters differ')
    need(base['selected_native_configuration']['ctr_target_border_count']=={
        'present_in_get_all_params':True,'value':1}, 'Baseline global count differs')
    for family in ('simple_ctr','combinations_ctr'):
        need(base['selected_native_configuration'][family]['present_in_get_all_params'] is True
             and base['selected_native_configuration'][family]['value']==feas['models']['mae_control']['saved_native_parameters'][family],
             'Original baseline descriptors differ from artificial control')
    op=base['operations']
    need(op['saved_model_loaded'] is True and op['model_methods_used']==['load_model','get_all_params']
         and all(op[k] is False for k in ('MAE_or_gates_recomputed','X_y_or_saved_prediction_arrays_read',
             'external_data_accessed','models_fitted','predictions_computed')), 'Baseline read-only scope differs')


def bind_baseline_release(bindings,base):
    for key in ('model','configuration','local_package_source'):bind_record(bindings,base[key])
    snap=base['released_snapshot_binding']
    bindings.add(snap['path'],snap['sha256']);bindings.add(snap['receipt_path'],snap['receipt_sha256'])
    snapshot=locked(safe(snap['path']))
    need(snapshot['version']=='v3' and snapshot['is_release'] is True, 'Wrong released snapshot')
    original=base['model']['path'];entry=snapshot['copied_files'][original]
    need(entry['original_sha256']==entry['backup_sha256']==base['model']['sha256']==snap['model_original_sha256']
         and entry['backup_path']==snap['model_backup_path'], 'Native model not bound to released snapshot')
    bindings.add(snap['model_backup_path'],snap['model_original_sha256'])
    v3=snapshot['copied_files']['outputs/submission_v3.csv']
    bindings.add('outputs/submission_v3.csv',v3['original_sha256'])
    bindings.add(v3['backup_path'],v3['backup_sha256'])
    v2=[v for v in snapshot['external_dependencies'] if v['original_path']=='outputs/submission_v2.csv']
    need(len(v2)==1, 'Missing protected v2 identity')
    bindings.add(v2[0]['original_path'],v2[0]['sha256'])
    for name in v2[0]['existing_backups']:bindings.add(name,v2[0]['sha256'])
    # Deliberately do not inherit the unrelated entire v3 snapshot dependency tree.


def validate_report(report,feas,review):
    need(report['round']==23 and report['status']=='research_not_promoted' and report['reason']==REASON
         and report['deployment_decision']=='keep_v3'
         and report['recommended_submission']=='outputs/submission_v3.csv'
         and report['new_submission_created'] is False and report['backtest_accepted'] is None, 'Wrong report conclusion')
    need(all(report[k] is False for k in ('formal_price_plan_created','price_training_performed','price_evaluation_performed')),
         'Report invents a price experiment')
    stages=report['price_stages']
    if isinstance(stages,list):stages={s['stage']:s for s in stages}
    need(set(stages)=={'development','confirmation','auxiliary','final'}, 'Missing price-stage absences')
    for stage in stages.values():
        need(stage['status']=='not_executed_research_not_promoted'
             and all(stage[k] is None for k in ('metrics','mae','weighted_mae','gate_results','fit_rows','query_rows')),
             'Unexecuted price stage must remain null')
    probe=report['synthetic_probe']
    need([probe[k] for k in ('fits_completed','models_saved','models_reloaded','fully_verified_models')]==[2,2,2,1],
         'Report artificial counts differ')
    for name,record in feas['models'].items():
        item=probe['models'][name]
        need(item['original_record_status']==record['status'], 'Report changes partial model status')
        need(all(item[k] is True for k in ('fit_completed','model_saved','model_reloaded'))
             and item['fully_verified'] is (name=='mae_control'), 'Report verification scope differs')
        for k in ('fitted_ctr_evidence','saved_ctr_evidence','model_path','model_sha256'):
            need(item[k]==record[k], 'Report original evidence changed: '+k)
    need(probe['models']['ctr8_candidate']['saved_reload_predictions_exact_recorded'] is None,
         'Candidate complete verification was not persisted')


def validate_qa(qa,report_sha,result_sha):
    need(qa['round']==23 and qa['status']=='passed' and qa['report_sha256']==report_sha
         and qa['report_results_sha256']==result_sha, 'QA report identity differs')
    need(qa['protected_unchanged'] is True and qa['protected_before']==qa['protected_after']
         and set(qa['protected_before'])=={'outputs/submission_v2.csv','outputs/submission_v3.csv','runs/versions.json'},
         'QA protected artifact contract differs')
    need(all(qa[k] is False for k in ('arrays_or_models_loaded','MAE_recomputed','gates_recalculated','formal_price_fit_performed')),
         'QA exceeded JSON/byte scope')
    need(isinstance(qa['checks'],list) and qa['checks'] and all(isinstance(k,str) for k in qa['checks'])
         and len(set(qa['checks']))==qa['check_count']==len(qa['checks']), 'QA check inventory invalid')


def context():
    return {name:{'sha256':sha(safe(name)),'bytes':safe(name).stat().st_size}
            for name in sorted(MUTABLE) if (ROOT/name).is_file()}


def collect():
    before=context();files=inventory();bindings=Bindings()
    need(read(safe('runs/versions.json'))['recommended_version']=='v3', 'Current registry must still recommend v3')
    feas=bindings.authority(FEAS);need(sha(safe(FEAS))==FIXED_FEAS_SHA,'Original partial research evidence changed')
    bindings.add(SOURCES[0],FIXED_RECIPE_SHA);validate_feasibility(feas)
    base=bindings.authority(BASELINE);validate_baseline(base,feas)
    history=bindings.authority(HISTORY)
    need(history['round']==23 and history['research_only'] is True
         and history['status']=='bounded_source_history_review_complete'
         and history['read_scope']['fitting_or_scoring_performed'] is False, 'History scope differs')
    review=bindings.authority(REVIEW);validate_review(review,feas)
    for value in (feas,base,history,review):bind_maps(bindings,value)
    for record in feas['models'].values():bindings.add(record['model_path'],record['model_sha256'])
    bindings.add(review['review_document'],review['review_document_sha256'])
    bind_baseline_release(bindings,base)
    for name in FEEDBACK:
        value=bindings.authority(name)
        bindings.add(value['screenshot_path'],value['screenshot_sha256'])
        if 'submission_path' in value:bindings.add(value['submission_path'],value['submission_sha256'])
    report=bindings.authority('reports/round23_results.json');validate_report(report,feas,review)
    bindings.mapping(report['source_input_sha256'])
    bindings.mapping(report['read_and_reverified_sha256']);bindings.mapping(report['inherited_dependency_sha256'])
    bindings.add(report['report_path'],report['report_sha256'])
    need(report['report_path']=='reports/round23_report.md', 'Wrong report path')
    qa=bindings.authority(QA_PATH);validate_qa(qa,report['report_sha256'],sha(safe('reports/round23_results.json')))
    for name,identity in qa['protected_after'].items():bind_record(bindings,{'path':name,**identity})
    for name in sorted(files):bindings.add(name)
    logname='reports/round23_work_log.md';lograw=safe(logname).read_bytes()
    need(closed_work_log_sha256(lograw)==bindings.values[logname]['sha256'], 'Work-log bytes changed')
    copied={n:bindings.values[n] for n in sorted(files)}
    need(not any(copy_scope(n) and n not in files for n in bindings.values), 'Owned dependency absent from inventory')
    external={n:v for n,v in sorted(bindings.values.items()) if n not in files}
    need(context()==before, 'Mutable context changed during collection')
    state={'round':23,'is_release':False,'research_status':'not_advanced_to_price_candidate',
        'reason':REASON,'formal_price_plan_created':False,'price_training_performed':False,
        'price_evaluation_performed':False,'backtest_accepted':None,'new_submission_created':False,
        'recommended_version':'v3','artificial_fits_completed':2,'artificial_models_saved_and_loaded':2,
        'fully_verified_models':1,'arrays_or_models_deserialized':False,
        'feasibility_sha256':copied[FEAS]['sha256'],'baseline_probe_sha256':copied[BASELINE]['sha256'],
        'history_sha256':copied[HISTORY]['sha256'],'independent_review_sha256':copied[REVIEW]['sha256'],
        'report_json_sha256':copied['reports/round23_results.json']['sha256'],
        'report_artifact_qa_sha256':copied[QA_PATH]['sha256'],
        'source_sha256':copied['src/archive_round23.py']['sha256'],
        'closed_work_log_sha256':copied[logname]['sha256'],
        'mutable_context_at_creation':before,'mutable_declarations_not_live_guards':{n:sorted(v) for n,v in bindings.mutable_declarations.items()}}
    return copied,external,state


def manifest_structure(manifest):
    need(manifest['kind']=='round23_research_not_promoted','Wrong snapshot kind')
    state=manifest['state']
    need(state['round']==23 and state['is_release'] is False
         and state['research_status']=='not_advanced_to_price_candidate' and state['reason']==REASON
         and state['backtest_accepted'] is None and state['recommended_version']=='v3'
         and all(state[k] is False for k in ('formal_price_plan_created','price_training_performed','price_evaluation_performed','new_submission_created')),
         'Snapshot invents a price experiment')
    need(state['artificial_fits_completed']==state['artificial_models_saved_and_loaded']==2
         and state['fully_verified_models']==1, 'Snapshot artificial counts differ')
    copied,external=manifest['copied_files'],manifest['external_dependencies']
    need(set(copied).isdisjoint(external),'Copy/external overlap')
    names=set(copied)|set(external)
    need(len({normalized(n).casefold() for n in names})==len(names),'Case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names}&{n.casefold() for n in MUTABLE},'Mutable permanent guard')
    need(set(SOURCES).union(DOCS,RESEARCH_FILES)<=set(copied) and all(copy_scope(n) for n in copied),'Wrong research inventory')
    need(all(not copy_scope(n) for n in external),'Owned artifact left external')
    need(manifest['copied_file_count']==len(copied) and manifest['external_dependency_count']==len(external)
         and manifest['copied_bytes']==sum(i['bytes'] for i in copied.values()), 'Wrong manifest counts')
    for name,item in {**copied,**external}.items():
        need(type(item['bytes']) is int and item['bytes']>=0 and re.fullmatch('[0-9a-f]{64}',item['sha256']), 'Invalid file identity')
    for key,name in [('feasibility_sha256',FEAS),('baseline_probe_sha256',BASELINE),('history_sha256',HISTORY),
        ('independent_review_sha256',REVIEW),('report_json_sha256','reports/round23_results.json'),
        ('report_artifact_qa_sha256',QA_PATH),('source_sha256','src/archive_round23.py'),
        ('closed_work_log_sha256','reports/round23_work_log.md')]:
        need(state[key]==copied[name]['sha256'],'Manifest authority differs: '+key)
    return copied,external



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
        need(inventory() == set(copied), 'Original R23 inventory changed after archive')
    return {'verified': True, 'snapshot_sha256': sha(folder / 'snapshot_manifest.json'),
            'copied_files': len(copied), 'external_dependencies': len(external),
            'copied_bytes': manifest['copied_bytes'], 'registry_and_current_status_not_live_guards': True,
            'labels_predictions_arrays_or_models_deserialized': False}

def create():
    need(not DEST.exists(), 'Snapshot exists; use --verify; never overwrite')
    need(not list(DEST.parent.glob('_round23_research_staging_*')),
         'Previous archive staging must be inspected and preserved')
    copied, external, state = collect()
    staging = DEST.parent / ('_round23_research_staging_' + uuid.uuid4().hex)
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
    manifest = {'kind': 'round23_research_not_promoted',
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

def work_log_self_check():
    checks = []
    closed = '<!-- WORK_LOG_STATUS: CLOSED -->'
    for name, raw in [('LF', ('# Log\n'+closed+'\n').encode('utf-8')),
                      ('UTF8_BOM_CRLF', ('\ufeff# 紀錄\r\n'+closed+'\r\n').encode('utf-8'))]:
        need(closed_work_log_sha256(raw) == hashlib.sha256(raw).hexdigest(), 'Work-log bytes SHA differs')
        checks.append('closed_marker_'+name+'_accepted_and_same_bytes_bound')
    cases = {
        'missing': '# Log\n',
        'open': '<!-- WORK_LOG_STATUS: OPEN -->\n',
        'duplicate': closed+'\n'+closed+'\n',
        'closed_plus_open': closed+'\n<!-- WORK_LOG_STATUS: OPEN -->\n',
        'indented': ' '+closed+'\n',
        'unknown_status': closed+'\n<!-- WORK_LOG_STATUS: UNKNOWN -->\n',
        'wrong_case': '<!-- work_log_status: CLOSED -->\n',
    }
    for name,text in cases.items():
        try:
            closed_work_log_sha256(text.encode('utf-8'))
        except RuntimeError:
            checks.append(name+'_rejected')
        else:
            raise AssertionError('Expected work-log rejection: '+name)
    return checks


def synthetic_fixture():
    native={'ctr_target_border_count':1,'iterations':2,'depth':8,'task_type':'GPU','loss_function':'MAE','eval_metric':'MAE',
            'simple_ctr':['Borders:CtrBorderCount=15:TargetBorderCount=1:Prior=0/1','FeatureFreq:CtrBorderCount=15:Prior=0/1'],
            'combinations_ctr':['Borders:CtrBorderCount=15:TargetBorderCount=1:Prior=0/1','FeatureFreq:CtrBorderCount=15:Prior=0/1']}
    params={'iterations':2999,'learning_rate':.065,'depth':8,'boosting_type':'Plain','thread_count':4,
            'loss_function':'MAE','eval_metric':'MAE','task_type':'GPU'}
    f={'round':23,'status':'synthetic_GPU_compatibility_failed','synthetic_only':True,
       'property_X_y_models_or_predictions_loaded':False,'property_MAE_computed':False,'query_labels_used_for_training':False,
       'planned_synthetic_model_count':2,'synthetic_model_count':1,'fit_rows':512,'query_rows':32,'feature_count':95,
       'categorical_feature_count':23,'synthetic_seed':20260925,'candidate_passed_parameter_changes':['ctr_target_border_count'],
       'declared_candidate_ctr_target_border_count':8,'feasibility_overrides':{'iterations':2,'verbose':False},
       'border_values_searched':False,'alternate_values_attempted':False,'no_alternative_value_or_retry_attempted':True,
       'baseline_parameters':params,'candidate_parameters':dict(params,ctr_target_border_count=8),
       'failure_phase':'verify_artificial_model:ctr8_candidate',
       'error':{'type':'ValueError','message':'Target-based native simple_ctr does not reflect the global target-border count'},
       'models':{},'source_sha256':{'src/example.py':'a'*64},'input_sha256':{'runs/config.json':'b'*64}}
    for name,count in [('mae_control',1),('ctr8_candidate',8)]:
        n=dict(native,ctr_target_border_count=count)
        row={'status':'passed' if count==1 else 'started','fit_rows':512,'query_rows':32,
             'passed_parameters':dict(f['baseline_parameters' if count==1 else 'candidate_parameters'],iterations=2,verbose=False),
             'model_path':'outputs/round23_research/'+name+'.cbm','model_sha256':'c'*64}
        for side in ('fitted','saved'):
            row[side+'_native_parameters']=copy.deepcopy(n);row[side+'_ctr_evidence']=ctr_evidence(n)
        if count==1:row.update(saved_reload_predictions_exact=True,tree_count=2,feature_count=95,categorical_positions=list(range(23)))
        f['models'][name]=row
    r={'round':23,'status':'review_completed','research_decision':'not_advanced_to_price_candidate','reason':REASON,
       'artificial_fits_completed':2,'artificial_models_saved_and_loaded':2,'fully_verified_models':1,
       'native_ctr_evidence':{n:{k:v[k] for k in ('fitted_ctr_evidence','saved_ctr_evidence')} for n,v in f['models'].items()},
       'artificial_model_artifacts_from_original_evidence':{n:{k:v[k] for k in ('model_path','model_sha256','status')} for n,v in f['models'].items()},
       'error':f['error'],'failure_phase':f['failure_phase'],'source_sha256':f['source_sha256'],'input_sha256':f['input_sha256'],
       'interpretation':{'parameter_accepted_and_artificial_fit_completed':True,**{k:False for k in (
           'gpu_parameter_unsupported_proven','all_underlying_ctr_computation_unchanged_proven',
           'intended_eight_target_borders_verified_in_native_descriptors','retry_attempted','alternate_value_attempted',
           'criterion_weakened','parser_misread_identified')}}}
    report={'round':23,'status':'research_not_promoted','reason':REASON,'deployment_decision':'keep_v3',
       'recommended_submission':'outputs/submission_v3.csv','new_submission_created':False,'backtest_accepted':None,
       'formal_price_plan_created':False,'price_training_performed':False,'price_evaluation_performed':False,
       'price_stages':[{'stage':k,'status':'not_executed_research_not_promoted',**{n:None for n in (
           'metrics','mae','weighted_mae','gate_results','fit_rows','query_rows')}} for k in ('development','confirmation','auxiliary','final')],
       'synthetic_probe':{'fits_completed':2,'models_saved':2,'models_reloaded':2,'fully_verified_models':1,'models':{}}}
    for n,v in f['models'].items():
        report['synthetic_probe']['models'][n]={'original_record_status':v['status'],
            'fit_completed':True,'model_saved':True,'model_reloaded':True,'fully_verified':n=='mae_control',
            'saved_reload_predictions_exact_recorded':True if n=='mae_control' else None,
            **{k:v[k] for k in ('fitted_ctr_evidence','saved_ctr_evidence','model_path','model_sha256')}}
    protected={k:{'sha256':'a'*64,'bytes':2} for k in ('outputs/submission_v2.csv','outputs/submission_v3.csv','runs/versions.json')}
    qa={'round':23,'status':'passed','report_sha256':'a'*64,'report_results_sha256':'b'*64,
        'protected_unchanged':True,'protected_before':protected,'protected_after':copy.deepcopy(protected),
        'arrays_or_models_loaded':False,'MAE_recomputed':False,'gates_recalculated':False,'formal_price_fit_performed':False,
        'checks':['one','two'],'check_count':2}
    return f,r,report,qa


def self_check():
    """In-memory contracts and source AST only: no real authorities or models."""
    checks=work_log_self_check()
    def good(name,action):action();checks.append(name)
    def bad(name,action):
        try:action()
        except (RuntimeError,KeyError,ValueError,TypeError):checks.append(name);return
        raise AssertionError('Expected refusal: '+name)
    def changed(value,path,replacement):
        value=copy.deepcopy(value);obj=value
        for key in path[:-1]:obj=obj[key]
        obj[path[-1]]=replacement;return value
    f,r,report,qa=synthetic_fixture()
    good('partial_candidate_native_mismatch_preserved',lambda:validate_feasibility(f))
    good('independent_review_scope_accurate',lambda:validate_review(r,f))
    good('research_report_all_price_stages_null',lambda:validate_report(report,f,r))
    good('small_json_qa_protected_hashes',lambda:validate_qa(qa,'a'*64,'b'*64))
    cases=[(['round'],22),(['status'],'synthetic_GPU_compatibility_passed'),(['synthetic_only'],False),
      (['property_X_y_models_or_predictions_loaded'],True),(['property_MAE_computed'],True),(['query_labels_used_for_training'],True),
      (['planned_synthetic_model_count'],3),(['synthetic_model_count'],2),(['fit_rows'],500),(['query_rows'],20),
      (['feature_count'],97),(['categorical_feature_count'],25),(['synthetic_seed'],42),
      (['declared_candidate_ctr_target_border_count'],16),(['candidate_passed_parameter_changes'],['one_hot_max_size']),
      (['border_values_searched'],True),(['alternate_values_attempted'],True),(['no_alternative_value_or_retry_attempted'],False),
      (['candidate_parameters','depth'],6),(['failure_phase'],'fit:ctr8_candidate'),(['error','type'],'CatBoostError'),
      (['models','ctr8_candidate','status'],'passed'),(['models','mae_control','saved_reload_predictions_exact'],False),
      (['models','ctr8_candidate','saved_native_parameters','ctr_target_border_count'],1)]
    for path,value in cases:
        bad('reject_feas_'+'.'.join(path),lambda path=path,value=value:validate_feasibility(changed(f,path,value)))
    extra=copy.deepcopy(f);extra['models']['ctr8_candidate']['saved_reload_predictions_exact']=True
    bad('reject_invented_candidate_success',lambda:validate_feasibility(extra))
    bad('reject_gpu_unsupported_overclaim',lambda:validate_review(changed(r,['interpretation','gpu_parameter_unsupported_proven'],True),f))
    bad('reject_underlying_algorithm_unchanged_overclaim',lambda:validate_review(changed(r,['interpretation','all_underlying_ctr_computation_unchanged_proven'],True),f))
    for key in ('formal_price_plan_created','price_training_performed','price_evaluation_performed','new_submission_created'):
        bad('reject_report_'+key,lambda key=key:validate_report(changed(report,[key],True),f,r))
    bad('reject_false_backtest_rejection',lambda:validate_report(changed(report,['backtest_accepted'],False),f,r))
    for field in ('metrics','mae','weighted_mae','gate_results','fit_rows','query_rows'):
        bad('reject_unrun_'+field,lambda field=field:validate_report(changed(report,['price_stages',0,field],0),f,r))
    bad('reject_fully_verified_two',lambda:validate_report(changed(report,['synthetic_probe','fully_verified_models'],2),f,r))
    bad('reject_candidate_reload_claim',lambda:validate_report(changed(report,['synthetic_probe','models','ctr8_candidate','saved_reload_predictions_exact_recorded'],True),f,r))
    for field in ('arrays_or_models_loaded','MAE_recomputed','gates_recalculated','formal_price_fit_performed'):
        bad('reject_QA_'+field,lambda field=field:validate_qa(changed(qa,[field],True),'a'*64,'b'*64))
    bad('reject_QA_report_identity',lambda:validate_qa(qa,'c'*64,'b'*64))
    bad('reject_QA_protected_changed',lambda:validate_qa(changed(qa,['protected_after','runs/versions.json','bytes'],3),'a'*64,'b'*64))
    bad('reject_QA_duplicate_checks',lambda:validate_qa(changed(qa,['checks'],['one','one']),'a'*64,'b'*64))
    n=f['models']['mae_control']['saved_native_parameters']
    need(ctr_evidence(n)['simple_ctr'][1]['target_border_count'] is None,'Absent is not null');checks.append('FeatureFreq_missing_count_is_null')
    bad('reject_duplicate_ctr_token',lambda:ctr_evidence(dict(n,simple_ctr=['Borders:TargetBorderCount=1:TargetBorderCount=8'])))
    for path in ('../escape','C:/escape','/absolute','dir//file','dir/./file','dir/../file','a\x00b'):
        bad('reject_unsafe_'+repr(path),lambda path=path:normalized(path))
    for path in ('.development/model.cbm','_stage_x/test.json','foo.tmp','folder/partial.part','a.lock'):
        bad('reject_current_staging_'+path,lambda path=path:validate_current_paths([path]))
    need(copy_scope('outputs/round23_research/ctr8_candidate.cbm') and not copy_scope('outputs/submission_v3.csv')
         and not copy_scope('runs/versions.json') and not copy_scope('src/round22_huber_geo.py'),'Copy scope differs')
    checks.append('owned_research_only_copy_scope')
    b=Bindings();need(b.declare('src/a.py','a'*64),'Declare');need(b.declare('src/a.py','a'*64),'Idempotent')
    b.values['src/a.py']={'sha256':'a'*64,'bytes':1}
    bad('reject_conflicting_SHA',lambda:b.declare('src/a.py','b'*64))
    bad('reject_case_ambiguous_SHA',lambda:b.declare('src/A.py','a'*64))
    bad('reject_malformed_SHA',lambda:b.declare('src/b.py','notsha'))
    need(b.declare('runs/versions.json','a'*64) is False and 'runs/versions.json' not in b.values,'Mutable live guard')
    checks.append('registry_context_not_permanent_guard')
    tree=ast.parse(Path(__file__).read_text(encoding='utf-8-sig'))
    imports={n.names[0].name for n in tree.body if isinstance(n,ast.Import)}
    need(not imports&{'numpy','pandas','catboost','pickle'},'Model/array import found');checks.append('stdlib_only_no_model_array_import')
    create_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='create')
    text=ast.get_source_segment(Path(__file__).read_text(encoding='utf-8-sig'),create_node)
    need(text.index('verify_at(staging)')<text.index('staging.rename(DEST)'),'Publish before verification')
    need("not DEST.exists()" in text and "open('xb')" in text and 'shutil.rmtree' not in text,'Nonoverwrite safety changed')
    checks.append('atomic_publish_after_verification_nonoverwrite_no_cleanup')
    need(len(set(checks))==len(checks),'Duplicate check names')
    return {'synthetic_only':True,'real_authorities_models_arrays_or_prices_read':False,'files_written':0,
            'checks':checks,'count':len(checks),'script_sha256':sha(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create',action='store_true');mode.add_argument('--verify',action='store_true')
    mode.add_argument('--self-check',action='store_true');args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    elif args.verify:print(json.dumps(verify_at(DEST),ensure_ascii=False),flush=True)
    else:create()
