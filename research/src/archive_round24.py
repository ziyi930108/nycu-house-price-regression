"""R24 experiment byte snapshot. No fitting, model/array loading or scoring.
Completed stages come from finalized receipts; this never publishes a release.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round24'
DEST=ROOT/'runs/round24_experiments'
STAGES=('development','confirmation','auxiliary')
SOURCES=('src/round24_seed_ensemble.py','src/round24_experiment.py',
         'src/round24_verify_composition.py','src/archive_round24.py')
DOCS=tuple('reports/round24_'+n for n in ('plan.md','method_review.md','pipeline_review.md',
           'work_log.md','report.md','results.json','results.sha256'))
PROOF='outputs/round24/audit/composition_verification.json'
AUTH=('outputs/round24/candidate_plan.json','outputs/round24/frozen_candidate.json',
      'outputs/round24/accepted_candidate.json',PROOF,'reports/round24_results.json',
      'outputs/round24_research/method_sources.json')
MUTABLE={'runs/versions.json','CURRENT_STATUS.md'}
NOTICE='# R24 experiment snapshot\n\nThis records only the finalized executed backtests. It is not a release and\ncreates no CSV or registry change. Seed42 is the saved R6 model; only fixed\nseeds137/2026 are newly fitted. Each branch is clipped to nonnegative TWD\nbefore the fixed equal mean replaces the existing geography slot.\n\nThe saved-output composition proof does not independently load CBMs. Full-query\nCBM save/reload checks are recorded by the model driver. This archiver validates\nreceipts and raw bytes only, never re-computing predictions or price metrics.\n\nOwned source/output/research/report files are copied. Original data, prior\nmodels and map/source dependencies remain external SHA/size requirements.\nMutable registry/status files are creation-time context, not permanent guards.\nCLOSED and complete finalized receipts are required. Existing snapshots are\nnever overwritten; interrupted staging is preserved for inspection.\n'


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
    markers=[line for line in raw.decode('utf-8-sig').splitlines() if 'work_log_status' in line.casefold()]
    need(markers==['<!-- WORK_LOG_STATUS: CLOSED -->'],'Work log must have exactly one CLOSED marker')
    return hashlib.sha256(raw).hexdigest()

def context():return {n:{'sha256':sha(ROOT/n),'bytes':(ROOT/n).stat().st_size} for n in sorted(MUTABLE) if (ROOT/n).is_file()}
def copy_scope(n):return n in SOURCES or n.startswith(('outputs/round24/','outputs/round24_research/','reports/round24_'))


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
    need(accepted['round']==24 and type(accepted['accepted']) is bool and accepted['new_submission_created'] is False
         and accepted['currently_recommended_submission']=='outputs/submission_v3.csv','Wrong finalization')
    done=set(accepted['stage_result_sha256'])
    need('development' in done and done<=set(STAGES),'Invalid completed stages')
    need(set(accepted['stage_status'])==set(STAGES),'Missing stage statuses')
    need({s for s,v in accepted['stage_status'].items() if v=='price_evaluated'}==done,'Completed stage status differs')
    need(set(accepted['unevaluated_stages'])==set(STAGES)-done,'Unevaluated stage inventory differs')
    if accepted['accepted']:need(done==set(STAGES),'An accepted backtest must complete all three stages')
    return [s for s in STAGES if s in done]


def inventory(stages):
    need(OUT.is_dir() and (ROOT/'outputs/round24_research').is_dir(),'Missing owned experiment/research')
    for part in ('models','evaluation'):
        folder=OUT/part;need(folder.is_dir() and {p.name for p in folder.iterdir()}==set(stages),'Partial/extra stages in '+part)
    need(not (OUT/'final').exists() and not (OUT/'release').exists(),'Final release is outside this snapshot')
    paths=set(OUT.rglob('*'))|set((ROOT/'outputs/round24_research').rglob('*'))|set((ROOT/'reports').glob('round24_*'))
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
    need(plan['round']==freeze['round']==24 and accepted['plan_sha256']==freeze['plan_sha256']==plan_sha
         and accepted['frozen_candidate_sha256']==freeze_sha,'Final authorities do not share a plan')
    for key in ('source_sha256','dependency_sha256','research_dependency_sha256','final_dependency_sha256'):b.mapping(plan[key])
    for mapping in plan['stage_dependency_sha256'].values():b.mapping(mapping)
    b.mapping(freeze['development_artifact_sha256']);b.mapping(accepted['source_sha256'])
    proof=b.authority(PROOF)
    need(proof['round']==24 and proof['status']=='passed'
         and proof['verification_level']=='saved_raw_output_composition_and_existing_metric_replay'
         and proof['plan_sha256']==plan_sha and proof['frozen_candidate_sha256']==freeze_sha
         and proof['accepted_candidate_sha256']==sha(ROOT/AUTH[2]) and set(proof['checked_stages'])==set(stages),
         'Missing or mismatched completed composition proof')
    b.mapping(proof['source_sha256']);b.mapping(proof['source_input_sha256'])
    for stage in stages:
        root=OUT/'models'/stage;meta=b.authority(rel(root/'metadata.json'))
        need(meta['round']==24 and meta['stage']==stage and meta['plan_sha256']==plan_sha,'Wrong model metadata')
        for name,digest in meta['artifact_sha256'].items():b.add(rel(root/normalized(name)),digest)
        actual={p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
        need(actual==set(meta['artifact_sha256'])|{'metadata.json','metadata.sha256'},'Incomplete model file inventory')
        result_path=OUT/'evaluation'/stage/'results.json';result=b.authority(rel(result_path))
        need(result['round']==24 and result['stage']==stage and result['plan_sha256']==plan_sha
             and sha(result_path)==accepted['stage_result_sha256'][stage],'Final result identity differs')
        b.mapping(result['input_artifact_sha256']);b.mapping(result['output_artifact_sha256'])
        evidence=proof['stage_evidence'][stage]
        need(evidence['result_sha256']==sha(result_path) and evidence['model_metadata_sha256']==sha(root/'metadata.json')
             and evidence['metrics_all_fields_matched'] is True and evidence['decision_matched'] is True,
             'Composition proof does not cover this saved result/model')
    report=b.authority('reports/round24_results.json')
    need(report['round']==24 and report['accepted_candidate_sha256']==sha(ROOT/AUTH[2])
         and set(report['checked_stages'])==set(stages) and report['accepted']==accepted['accepted']
         and report['deployment_decision']==accepted['deployment_decision'],'Report finalization differs')
    b.add('reports/round24_report.md',report['report_sha256'])
    b.mapping(report['source_input_sha256'])
    research=b.authority('outputs/round24_research/method_sources.json')
    for key in ('source_sha256','evidence_sha256'):b.mapping(research[key])
    for n in sorted(files):b.add(n)
    log='reports/round24_work_log.md';need(closed((ROOT/log).read_bytes())==b.values[log]['sha256'],'CLOSED bytes changed')
    need(not any(copy_scope(n) and n not in files for n in b.values),'Owned dependency omitted')
    copied={n:b.values[n] for n in sorted(files)};external={n:v for n,v in sorted(b.values.items()) if n not in files}
    need(context()==before,'Mutable context changed while collecting')
    state={'round':24,'is_release':False,'accepted_backtest':accepted['accepted'],'checked_stages':stages,
           'deployment_decision':accepted['deployment_decision'],'new_submission_created':False,
           'recommended_version':'v3','mutable_context_at_creation':before,
           'authority_sha256':{n:copied[n]['sha256'] for n in AUTH},'closed_work_log_sha256':copied[log]['sha256'],
           'archive_source_sha256':copied['src/archive_round24.py']['sha256']}
    return copied,external,state


def verify_at(folder):
    folder=Path(folder).resolve();need(folder.is_relative_to((ROOT/'runs').resolve()),'Snapshot outside runs')
    m=locked(folder/'snapshot_manifest.json');need(m['kind']=='round24_experiment_snapshot','Wrong snapshot kind')
    c,e,state=m['copied_files'],m['external_dependencies'],m['state']
    need(state['round']==24 and state['is_release'] is False and state['new_submission_created'] is False,'Not an experiment snapshot')
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
    need(c['reports/round24_work_log.md']['sha256']==state['closed_work_log_sha256']
         and c['src/archive_round24.py']['sha256']==state['archive_source_sha256'],'Snapshot source/log identity differs')
    need((folder/'SNAPSHOT_README.md').read_bytes()==NOTICE.encode('utf-8'),'Notice differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==expected,'Extra/missing snapshot files')
    return {'status':'verified','snapshot_sha256':sha(folder/'snapshot_manifest.json'),'copied_files':len(c),
            'external_dependencies':len(e),'checked_stages':state['checked_stages'],'models_or_arrays_loaded':False}


def create():
    need(not DEST.exists(),'Snapshot exists; never overwrite')
    need(not list(DEST.parent.glob('_round24_experiments_staging_*')),'Previous staging must be preserved and inspected')
    copied,external,state=collect();stage=DEST.parent/('_round24_experiments_staging_'+uuid.uuid4().hex)
    stage.mkdir(parents=True,exist_ok=False)
    for name,item in copied.items():
        target=safe('files/'+name,stage,False);need(not target.exists(),'Copy destination exists')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(safe(name),target)
        need(sha(target)==sha(safe(name))==item['sha256'] and target.stat().st_size==item['bytes'],'Copy/source changed')
    (stage/'SNAPSHOT_README.md').write_bytes(NOTICE.encode('utf-8'))
    m={'kind':'round24_experiment_snapshot','created_at_utc':datetime.now(timezone.utc).isoformat(),'state':state,
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
    raw=b'# Log\n<!-- WORK_LOG_STATUS: CLOSED -->\n';need(closed(raw)==hashlib.sha256(raw).hexdigest(),'CLOSED bytes SHA');checks.append('closed_same_bytes')
    for raw in (b'OPEN',b'<!-- WORK_LOG_STATUS: OPEN -->',b'<!-- WORK_LOG_STATUS: CLOSED -->\n<!-- WORK_LOG_STATUS: CLOSED -->'):
        reject('invalid_closed_'+str(len(raw)),lambda raw=raw:closed(raw))
    a={'round':24,'accepted':False,'new_submission_created':False,'currently_recommended_submission':'outputs/submission_v3.csv',
       'stage_result_sha256':{'development':'a'*64},'stage_status':{'development':'price_evaluated','confirmation':'not_executed_after_development_rejection','auxiliary':'not_executed_after_development_rejection'},'unevaluated_stages':['confirmation','auxiliary']}
    need(completed(a)==['development'],'Dev-only inventory');checks.append('unrun_stages_not_required')
    all_done=dict(a,stage_result_sha256={n:'a'*64 for n in STAGES},stage_status={n:'price_evaluated' for n in STAGES},unevaluated_stages=[])
    need(completed(all_done)==list(STAGES) and completed(dict(all_done,accepted=True))==list(STAGES),'Three-stage accepted/veto');checks.append('three_stage_accepted_or_veto')
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
