"""R26 experiment byte snapshot. No fitting, model/array loading or scoring.
Completed stages come from finalized receipts; this never publishes a release.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round26'
DEST=ROOT/'runs/round26_experiments'
STAGES=('development','confirmation','auxiliary')
SOURCES=('src/round26_intersection_features.py','src/round26_support_audit.py','src/round26_intersection_geo.py','src/round26_experiment.py','src/archive_round26.py')
DOCS=tuple('reports/round26_'+n for n in ('research_plan.md','address_support.md','plan.md','work_log.md','report.md','results.json','results.sha256'))
AUTH=('outputs/round26/candidate_plan.json','outputs/round26/frozen_candidate.json',
      'outputs/round26/accepted_candidate.json','reports/round26_results.json')
MUTABLE={'runs/versions.json','CURRENT_STATUS.md'}
NOTICE='# R26 experiment snapshot\n\nThis records the finalized development rejection of the fixed explicit road-pair\ngeographic CatBoost experiment. It is not a release and creates no CSV or\nregistry change. Confirmation, auxiliary and final training were not executed.\n\nFull-query raw/TWD save/reload checks and the original v3 control replay are\nrecorded by the model driver. No additional independent model or composition\nreplay is claimed. This archiver checks receipts and bytes only; it does not\nload models/arrays, re-compute predictions or decide price gates.\n\nOwned sources, outputs and reports are copied. Original data, old models and\nall annual map/source dependencies remain external SHA/size requirements.\nMutable registry/status files are creation-time context, not permanent guards.\nCLOSED and complete finalized receipts are required. Existing snapshots are\nnever overwritten; interrupted staging is preserved for inspection.\n'


def need(ok,message):
    if not ok:raise RuntimeError(message)

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()

def encoded(value):return (json.dumps(value,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def normalized(name):
    need(isinstance(name,str),'Expected relative path');name=name.replace('\\','/')
    need(name and name==name.strip() and ':' not in name and '\x00' not in name
         and not PurePosixPath(name).is_absolute() and all(p not in ('','.','..') for p in name.split('/')),'Unsafe path: '+repr(name))
    return name

def safe(name,base=ROOT,exists=True):
    path=(Path(base)/normalized(name)).resolve();need(path.is_relative_to(Path(base).resolve()),'Path escaped root')
    if exists:need(path.is_file(),'Missing file: '+str(path))
    return path

def rel(path):return Path(path).resolve().relative_to(ROOT.resolve()).as_posix()
def locked(path):
    path=Path(path);receipt=path.with_suffix('.sha256')
    need(receipt.is_file() and receipt.read_text(encoding='ascii').strip()==sha(path),'Missing/changed receipt: '+str(path))
    return read(path)

def closed(raw):
    markers=[line for line in raw.decode('utf-8-sig').splitlines() if '<!-- CLOSED -->' in line]
    need(markers==['<!-- CLOSED -->'],'Work log must have exactly one CLOSED marker')
    return hashlib.sha256(raw).hexdigest()

def context():return {n:{'sha256':sha(ROOT/n),'bytes':(ROOT/n).stat().st_size} for n in sorted(MUTABLE) if (ROOT/n).is_file()}
def copy_scope(n):return n in SOURCES or n.startswith(('outputs/round26/','outputs/round26_research/','reports/round26_'))


class Bindings:
    def __init__(self):self.values={};self.cases={}
    def add(self,name,digest=None):
        name=normalized(name)
        need(self.cases.get(name.casefold(),name)==name,'Case-ambiguous input');self.cases[name.casefold()]=name
        if name.casefold() in {n.casefold() for n in MUTABLE}:
            need(name in MUTABLE,'Case-ambiguous mutable path');return None
        if digest is not None:need(isinstance(digest,str) and re.fullmatch('[0-9a-f]{64}',digest),'Invalid SHA')
        if name in self.values:
            need(digest is None or self.values[name]['sha256']==digest,'Conflicting SHA: '+name);return safe(name)
        path=safe(name);actual=sha(path);need(digest is None or digest==actual,'Changed input: '+name)
        self.values[name]={'sha256':actual,'bytes':path.stat().st_size};return path
    def mapping(self,mapping):
        need(isinstance(mapping,dict),'Expected path-to-SHA map')
        for name,digest in mapping.items():self.add(name,digest)
    def authority(self,name):
        path=self.add(name);need(path is not None,'Mutable authority prohibited')
        value=locked(path);self.add(rel(path.with_suffix('.sha256')));return value


def completed(accepted):
    need(accepted['round']==26 and accepted['accepted'] is False and accepted['new_submission_created'] is False
         and accepted['currently_recommended_submission']=='outputs/submission_v3.csv'
         and accepted['deployment_decision']=='keep_v3'
         and accepted['stopping_reason']=='development_price_gates_failed'
         and accepted['selected_candidate']=='v3_control','Only finalized development rejection is supported')
    need(set(accepted['stage_result_sha256'])=={'development'},'Wrong completed stages')
    need(accepted['stage_status']=={'development':'price_evaluated',
         'confirmation':'not_executed_after_development_rejection',
         'auxiliary':'not_executed_after_development_rejection'},'Wrong stage statuses')
    need(set(accepted['unevaluated_stages'])=={'confirmation','auxiliary'},'Wrong unexecuted stages')
    return ['development']


def inventory(stages):
    need(OUT.is_dir(),'Missing owned experiment')
    for part in ('models','evaluation'):
        folder=OUT/part;need(folder.is_dir() and {p.name for p in folder.iterdir()}==set(stages),'Partial/extra stages in '+part)
    need(not (OUT/'final').exists() and not (OUT/'release').exists(),'Final release is outside this snapshot')
    research=ROOT/'outputs/round26_research';need(research.is_dir(),'Missing completed pure-X research')
    paths=set(OUT.rglob('*'))|set(research.rglob('*'))|set((ROOT/'reports').glob('round26_*'))
    paths.update(ROOT/n for n in SOURCES)
    files=set()
    for path in paths:
        need(not path.is_symlink(),'Symlink not an artifact');n=rel(path)
        need(not any(p.startswith(('.', '_stage_')) for p in n.split('/'))
             and not n.lower().endswith(('.tmp','.part','.lock')),'Incomplete staging: '+n)
        if path.is_file():
            need(copy_scope(n) and n not in MUTABLE,'Artifact outside owned scope')
            need(not(path.name.startswith('submission') and path.suffix.lower()=='.csv'),'No new CSV in experiment')
            files.add(n)
    required=set(SOURCES)|set(DOCS)|{p for n in AUTH for p in (n,n[:-5]+'.sha256')}
    need(required<=files,'Required source/report/receipt missing')
    return files


def collect():
    before=context();b=Bindings();plan=b.authority(AUTH[0]);freeze=b.authority(AUTH[1]);accepted=b.authority(AUTH[2])
    stages=completed(accepted);files=inventory(stages);plan_sha=sha(ROOT/AUTH[0]);freeze_sha=sha(ROOT/AUTH[1])
    need(plan['round']==freeze['round']==26 and accepted['plan_sha256']==freeze['plan_sha256']==plan_sha
         and accepted['frozen_candidate_sha256']==freeze_sha,'Final authorities do not share a plan')
    need(plan['candidate']==accepted['candidate']==freeze['candidate']=='intersection_geography_blend'
         and freeze['passed'] is False and freeze['later_stages_allowed'] is False
         and freeze['selected_candidate']=='v3_control','Wrong saved development selection')
    for key in ('source_sha256','dependency_sha256','research_dependency_sha256','final_dependency_sha256'):b.mapping(plan[key])
    need(bool(plan['research_dependency_sha256']),'Missing frozen pure-X research bindings')
    for mapping in plan['stage_dependency_sha256'].values():b.mapping(mapping)
    b.mapping(freeze['development_artifact_sha256']);b.mapping(accepted['source_sha256'])
    root=OUT/'models/development';meta=b.authority(rel(root/'metadata.json'))
    need(meta['round']==26 and meta['stage']=='development' and meta['plan_sha256']==plan_sha
         and meta['source_sha256']==plan['source_sha256'],'Wrong model metadata')
    need(meta['reload_all_query_rows_verified'] is True and meta['reload_raw_all_query_rows_verified'] is True
         and meta['reload_query_rows']==meta['query_rows'],'Driver full-query reload evidence absent')
    for name,digest in meta['artifact_sha256'].items():b.add(rel(root/normalized(name)),digest)
    actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    need(actual==set(meta['artifact_sha256'])|{'metadata.json','metadata.sha256'},'Incomplete model inventory')
    result_path=OUT/'evaluation/development/results.json';result=b.authority(rel(result_path))
    need(result['round']==26 and result['stage']=='development' and result['plan_sha256']==plan_sha
         and result['source_sha256']==plan['source_sha256']
         and sha(result_path)==accepted['stage_result_sha256']['development']==freeze['result_sha256'],
         'Final result identity differs')
    need(result['decision']['passed'] is False and result['decision']['selected_candidate']=='v3_control'
         and result['v3_control_replayed_all_rows'] is True
         and result['new_model_replayed_all_rows'] is True
         and result['new_model_raw_outputs_replayed_all_rows'] is True,'Saved decision/replay evidence differs')
    b.mapping(result['input_artifact_sha256']);b.mapping(result['output_artifact_sha256'])
    report=b.authority('reports/round26_results.json')
    need(report['round']==26 and report['accepted_candidate_sha256']==sha(ROOT/AUTH[2])
         and report['checked_stages']==stages and report['accepted'] is False
         and report['deployment_decision']=='keep_v3','Report finalization differs')
    need(report['stages']['development']['metrics']==result['metrics']
         and report['stages']['development']['decision']==result['decision'],'Report saved development values differ')
    for stage in ('confirmation','auxiliary'):
        need(report['stages'][stage]['metrics'] is None and report['stages'][stage]['decision'] is None,
             'Unexecuted report stage must have null results')
    b.add('reports/round26_report.md',report['report_sha256']);b.mapping(report['source_input_sha256'])
    for n in sorted(files):b.add(n)
    log='reports/round26_work_log.md';need(closed((ROOT/log).read_bytes())==b.values[log]['sha256'],'CLOSED bytes changed')
    need(not any(copy_scope(n) and n not in files for n in b.values),'Owned dependency omitted')
    copied={n:b.values[n] for n in sorted(files)};external={n:v for n,v in sorted(b.values.items()) if n not in files}
    need(context()==before,'Mutable context changed while collecting')
    state={'round':26,'is_release':False,'accepted_backtest':False,'checked_stages':stages,
           'deployment_decision':'keep_v3','new_submission_created':False,'recommended_version':'v3',
           'independent_model_replay_performed':False,'driver_full_query_reload_evidence_recorded':True,
           'mutable_context_at_creation':before,
           'authority_sha256':{n:copied[n]['sha256'] for n in AUTH},'closed_work_log_sha256':copied[log]['sha256'],
           'archive_source_sha256':copied['src/archive_round26.py']['sha256']}
    return copied,external,state


def verify_at(folder):
    folder=Path(folder).resolve();need(folder.is_relative_to((ROOT/'runs').resolve()),'Snapshot outside runs')
    m=locked(folder/'snapshot_manifest.json');need(m['kind']=='round26_experiment_snapshot','Wrong snapshot kind')
    c,e,state=m['copied_files'],m['external_dependencies'],m['state']
    need(state['round']==26 and state['is_release'] is False and state['new_submission_created'] is False
         and state['accepted_backtest'] is False and state['checked_stages']==['development']
         and state['deployment_decision']=='keep_v3','Not a development-rejection experiment snapshot')
    names=list(c)+list(e)
    need(len({n.casefold() for n in names})==len(names),'Duplicate/case-ambiguous snapshot paths')
    need(not {n.casefold() for n in names}&{n.casefold() for n in MUTABLE},'Mutable permanent dependency')
    need(set(SOURCES)|set(DOCS)<=set(c) and all(copy_scope(n) for n in c) and all(not copy_scope(n) for n in e),'Wrong copy scope')
    need(inventory(state['checked_stages'])==set(c),'Original artifact inventory changed')
    need(m['copied_file_count']==len(c) and m['external_dependency_count']==len(e)
         and m['copied_bytes']==sum(v['bytes'] for v in c.values()),'Snapshot counts differ')
    expected={'snapshot_manifest.json','snapshot_manifest.sha256','SNAPSHOT_README.md'}
    for name,item in {**c,**e}.items():
        original=safe(name);need(sha(original)==item['sha256'] and original.stat().st_size==item['bytes'],'Original dependency changed: '+name)
        if name in c:
            backup=safe('files/'+name,folder);need(sha(backup)==item['sha256'] and backup.stat().st_size==item['bytes'],'Snapshot bytes differ')
            expected.add('files/'+name)
    for name,digest in state['authority_sha256'].items():need(c[name]['sha256']==digest,'Snapshot authority changed')
    need(c['reports/round26_work_log.md']['sha256']==state['closed_work_log_sha256']
         and c['src/archive_round26.py']['sha256']==state['archive_source_sha256'],'Snapshot source/log identity differs')
    need((folder/'SNAPSHOT_README.md').read_bytes()==NOTICE.encode('utf-8'),'Notice differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==expected,'Extra/missing snapshot files')
    return {'status':'verified','snapshot_sha256':sha(folder/'snapshot_manifest.json'),'copied_files':len(c),
            'external_dependencies':len(e),'checked_stages':state['checked_stages'],'models_or_arrays_loaded':False}


def create():
    need(not DEST.exists(),'Snapshot exists; never overwrite')
    need(not list(DEST.parent.glob('_round26_experiments_staging_*')),'Previous staging must be preserved and inspected')
    copied,external,state=collect();stage=DEST.parent/('_round26_experiments_staging_'+uuid.uuid4().hex)
    stage.mkdir(parents=True,exist_ok=False)
    for name,item in copied.items():
        target=safe('files/'+name,stage,False);need(not target.exists(),'Copy destination exists')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(safe(name),target)
        need(sha(target)==sha(safe(name))==item['sha256'] and target.stat().st_size==item['bytes'],'Copy/source changed')
    (stage/'SNAPSHOT_README.md').write_bytes(NOTICE.encode('utf-8'))
    m={'kind':'round26_experiment_snapshot','created_at_utc':datetime.now(timezone.utc).isoformat(),'state':state,
       'copied_files':copied,'external_dependencies':external,'copied_file_count':len(copied),
       'external_dependency_count':len(external),'copied_bytes':sum(v['bytes'] for v in copied.values()),'self_contained':False}
    path=stage/'snapshot_manifest.json'
    with path.open('xb') as f:f.write(encoded(m))
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')
    verify_at(stage);need(context()==state['mutable_context_at_creation'],'Mutable context changed during copy')
    need(not DEST.exists(),'Concurrent destination appeared; preserve staging');stage.rename(DEST)
    print(json.dumps({'status':'created','snapshot_sha256':sha(DEST/'snapshot_manifest.json'),
          'copied_files':len(copied),'copied_bytes':m['copied_bytes'],'external_dependencies':len(external),
          'checked_stages':state['checked_stages']},ensure_ascii=False),flush=True)


def self_check():
    checks=[]
    def reject(name,fn):
        try:fn()
        except (RuntimeError,ValueError,KeyError):checks.append(name);return
        raise AssertionError('Expected refusal: '+name)
    for n in ('../outside','/absolute','C:/outside','a//b','a/../b'):
        reject('unsafe_'+n,lambda n=n:normalized(n))
    raw=b'# Log\n<!-- CLOSED -->\n';need(closed(raw)==hashlib.sha256(raw).hexdigest(),'CLOSED bytes SHA');checks.append('closed_same_bytes')
    for raw in (b'OPEN',b'<!-- OPEN -->',b'<!-- CLOSED -->\n<!-- CLOSED -->'):
        reject('invalid_closed_'+str(len(raw)),lambda raw=raw:closed(raw))
    a={'round':26,'accepted':False,'new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv',
       'deployment_decision':'keep_v3','stopping_reason':'development_price_gates_failed','selected_candidate':'v3_control',
       'stage_result_sha256':{'development':'a'*64},'stage_status':{'development':'price_evaluated','confirmation':'not_executed_after_development_rejection','auxiliary':'not_executed_after_development_rejection'},'unevaluated_stages':['confirmation','auxiliary']}
    need(completed(a)==['development'],'Dev-only inventory');checks.append('unrun_stages_not_required')
    all_done=dict(a,stage_result_sha256={n:'a'*64 for n in STAGES},stage_status={n:'price_evaluated' for n in STAGES},unevaluated_stages=[])
    reject('three_stage_scope_not_supported',lambda:completed(all_done))
    reject('accepted_incomplete',lambda:completed(dict(a,accepted=True)))
    need(not copy_scope('runs/versions.json') and not copy_scope('CURRENT_STATUS.md'),'Mutable copy scope');checks.append('mutable_context_only')
    b=Bindings();need(b.add('runs/versions.json','a'*64) is None,'Registry became guard');checks.append('registry_not_hashed_or_bound')
    reject('case_alias_mutable',lambda:Bindings().add('RUNS/versions.json','a'*64))
    return {'synthetic_only':True,'real_artifacts_read':False,'files_written':0,'checks':checks,'count':len(checks),'source_sha256':sha(__file__)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create',action='store_true');mode.add_argument('--verify',action='store_true');mode.add_argument('--self-check',action='store_true')
    args=parser.parse_args()
    if args.self_check:print(json.dumps(self_check(),ensure_ascii=False),flush=True)
    elif args.verify:print(json.dumps(verify_at(DEST),ensure_ascii=False),flush=True)
    else:create()
