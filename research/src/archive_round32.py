"""R32 experiment byte snapshot. No fitting, model/array loading or scoring.
Completed stages come from finalized receipts; this never publishes a release.
"""
from __future__ import annotations
import argparse, hashlib, json, re, shutil, uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/round32'
DEST=ROOT/'runs/round32_experiments'
STAGES=('development','confirmation','auxiliary')
SOURCES=('src/round32_xgboost_geo.py','src/round32_experiment.py','src/xgboost_geo_features.py','src/archive_round32.py')
BIG_MODEL='outputs/round32/models/development/model.ubj'
BIG_MODEL_BYTES=9831575601
DOCS=tuple('reports/round32_'+n for n in ('plan.md','work_log.md','report.md','results.json','results.sha256'))+('reports/xgboost_feasibility_research.md','reports/xgboost_feasibility_report.md')
AUTH=('outputs/round32/candidate_plan.json','outputs/round32/frozen_candidate.json',
      'outputs/round32/accepted_candidate.json','reports/round32_results.json')
MUTABLE={'runs/versions.json','CURRENT_STATUS.md'}
NOTICE='# R32 experiment snapshot\n\nFinalized development rejection of one fixed XGBoost raw geographic model.\nNo later-stage/final training, new CSV, release or registry change.\nDriver full-query raw/TWD UBJSON reload and evaluation control replay were\nrecorded; no extra independent native replay is claimed here. Byte checks only.\n\nSources, metadata, arrays, execution and reports are copied. The 9,831,575,601\nbyte development model.ubj deliberately remains an external immutable reference.\nIts SHA comes from saved metadata/result/freeze at creation (size checked);\nthe separate fresh --verify process hashes its original bytes exactly once.\nOther original data/models/map dependencies also remain external SHA/size guards.\nThe snapshot is NOT self-contained. All referenced original paths must remain.\nMutable registry/status are creation-time context only. CLOSED and finalized\nreceipts are required; existing snapshots are never overwritten. Failed staging\nis retained. No models/arrays are deserialized and no price scores recomputed.\n'



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
def copy_scope(n):return n!=BIG_MODEL and (n in SOURCES or n in DOCS or n.startswith(('outputs/round32/','reports/round32_')))


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
        path=safe(name)
        if name==BIG_MODEL:
            need(digest is not None and path.stat().st_size==BIG_MODEL_BYTES,'Missing large-model reference/size')
            actual=digest  # saved metadata authority; fresh --verify hashes this original once
        else:
            actual=sha(path);need(digest is None or digest==actual,'Changed input: '+name)
        self.values[name]={'sha256':actual,'bytes':path.stat().st_size};return path
    def mapping(self,mapping):
        need(isinstance(mapping,dict),'Expected path-to-SHA map')
        for name,digest in mapping.items():self.add(name,digest)
    def authority(self,name):
        path=self.add(name);need(path is not None,'Mutable authority prohibited')
        value=locked(path);self.add(rel(path.with_suffix('.sha256')));return value


def completed(accepted):
    need(accepted['round']==32 and accepted['accepted'] is False
         and accepted['final_training_allowed'] is False and accepted['new_CSV_created'] is False
         and accepted['registry_modified'] is False and accepted['selected_candidate']=='v3_control',
         'Only finalized development rejection is supported')
    need(set(accepted['decisions'])=={'development'} and accepted['decisions']['development']['passed'] is False,
         'Wrong completed stages')
    need(set(accepted['evidence_sha256'])=={'outputs/round32/evaluation/development/results.json'},'Wrong result evidence')
    return ['development']


def inventory(stages):
    need(OUT.is_dir(),'Missing owned experiment')
    for part in ('models','evaluation'):
        folder=OUT/part;need(folder.is_dir() and {p.name for p in folder.iterdir()}==set(stages),'Partial/extra stages in '+part)
    need(not (OUT/'final').exists() and not (OUT/'release').exists(),'Final release is outside this snapshot')
    paths=set(OUT.rglob('*'))|set((ROOT/'reports').glob('round32_*'))
    paths.update(ROOT/n for n in (*SOURCES,*DOCS))
    files=set()
    for path in paths:
        need(not path.is_symlink(),'Symlink not an artifact');n=rel(path)
        need(not any(p.startswith(('.', '_stage_')) for p in n.split('/'))
             and not n.lower().endswith(('.tmp','.part','.lock')),'Incomplete staging: '+n)
        if path.is_file():
            if n==BIG_MODEL:continue
            need(copy_scope(n) and n not in MUTABLE,'Artifact outside owned scope')
            need(not(path.name.startswith('submission') and path.suffix.lower()=='.csv'),'No new CSV in experiment')
            files.add(n)
    required=set(SOURCES)|set(DOCS)|{p for n in AUTH for p in (n,n[:-5]+'.sha256')}
    need(required<=files,'Required source/report/receipt missing')
    return files


def collect():
    before=context();b=Bindings();plan=b.authority(AUTH[0]);freeze=b.authority(AUTH[1]);accepted=b.authority(AUTH[2])
    stages=completed(accepted);files=inventory(stages);plan_sha=sha(ROOT/AUTH[0])
    need(plan['round']==freeze['round']==32 and accepted['plan_sha256']==freeze['plan_sha256']==plan_sha,
         'Final authorities do not share a plan')
    need(plan['candidate']=='xgboost_geography_blend' and freeze['passed'] is False
         and freeze['selected_candidate']=='v3_control','Wrong frozen selection')
    for key in ('source_sha256','dependency_sha256','research_dependency_sha256','final_dependency_sha256'):
        b.mapping(plan.get(key,{}))
    for mapping in plan['stage_dependency_sha256'].values():b.mapping(mapping)
    b.mapping(freeze['evidence_sha256']);b.mapping(accepted['evidence_sha256'])
    root=OUT/'models/development';meta=b.authority(rel(root/'metadata.json'))
    need(meta['round']==32 and meta['stage']=='development' and meta['plan_sha256']==plan_sha
         and meta['source_sha256']==plan['source_sha256'],'Wrong model metadata')
    need(meta['reload_all_query_raw_and_TWD_bitwise_equal'] is True,'Driver all-query reload absent')
    b.mapping(meta['artifact_sha256'])
    actual={rel(p) for p in root.rglob('*') if p.is_file()}
    need(actual==set(meta['artifact_sha256'])|{rel(root/'metadata.json'),rel(root/'metadata.sha256')},'Incomplete model inventory')
    result_path=OUT/'evaluation/development/results.json';result=b.authority(rel(result_path))
    need(result['round']==32 and result['stage']=='development' and result['plan_sha256']==plan_sha
         and result['source_sha256']==plan['source_sha256']
         and sha(result_path)==accepted['evidence_sha256'][rel(result_path)],'Result identity differs')
    need(result['decision']==accepted['decisions']['development'] and result['decision']['passed'] is False
         and result['v3_control_all_rows_bitwise_replayed'] is True,'Saved decision/control evidence differs')
    b.mapping(result['input_artifact_sha256']);b.mapping(result['output_artifact_sha256'])
    report=b.authority('reports/round32_results.json')
    need(report['round']==32 and report['accepted_candidate_sha256']==sha(ROOT/AUTH[2])
         and report['checked_stages']==stages and report['accepted'] is False
         and report['deployment_decision']=='keep_v3' and report['new_submission_created'] is False,'Report finalization differs')
    need(report['stages']['development']['metrics']==result['metrics']
         and report['stages']['development']['decision']==result['decision'],'Report saved values differ')
    for stage in ('confirmation','auxiliary','final'):
        need(report['stages'][stage]['metrics'] is None and report['stages'][stage]['decision'] is None,
             'Unexecuted stage must have null values')
    need(report['model_external_reference']['path']==BIG_MODEL
         and report['model_external_reference']['sha256']==b.values[BIG_MODEL]['sha256']
         and report['model_external_reference']['bytes']==BIG_MODEL_BYTES
         and report['model_external_reference']['archive_copy'] is False,'Large model reference differs')
    b.add('reports/round32_report.md',report['report_sha256']);b.mapping(report['source_input_sha256'])
    need(report['protected_before']==report['protected_after'] and report['protected_unchanged'] is True,'Protected report differs')
    for name,item in report['protected_before'].items():
        if name not in MUTABLE:b.add(name,item['sha256'])
    for n in sorted(files):b.add(n)
    log='reports/round32_work_log.md';need(closed((ROOT/log).read_bytes())==b.values[log]['sha256'],'CLOSED bytes changed')
    need(not any(copy_scope(n) and n not in files for n in b.values),'Owned dependency omitted')
    copied={n:b.values[n] for n in sorted(files)};external={n:v for n,v in sorted(b.values.items()) if n not in files}
    need(BIG_MODEL in external and BIG_MODEL not in copied,'Large model must remain external')
    need(context()==before,'Mutable context changed while collecting')
    state={'round':32,'is_release':False,'accepted_backtest':False,'checked_stages':stages,
           'deployment_decision':'keep_v3','new_submission_created':False,'recommended_version':'v3',
           'independent_model_replay_performed':False,'driver_full_query_reload_evidence_recorded':True,
           'large_model_external_reference':{'path':BIG_MODEL,**external[BIG_MODEL]},
           'large_model_bytes_rehashed_during_create':False,'fresh_verify_required':True,
           'mutable_context_at_creation':before,
           'authority_sha256':{n:copied[n]['sha256'] for n in AUTH},'closed_work_log_sha256':copied[log]['sha256'],
           'archive_source_sha256':copied['src/archive_round32.py']['sha256']}
    return copied,external,state


def verify_at(folder):
    folder=Path(folder).resolve();need(folder.is_relative_to((ROOT/'runs').resolve()),'Snapshot outside runs')
    m=locked(folder/'snapshot_manifest.json');need(m['kind']=='round32_experiment_snapshot','Wrong snapshot kind')
    c,e,state=m['copied_files'],m['external_dependencies'],m['state']
    need(state['round']==32 and state['is_release'] is False and state['new_submission_created'] is False
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
    need(c['reports/round32_work_log.md']['sha256']==state['closed_work_log_sha256']
         and c['src/archive_round32.py']['sha256']==state['archive_source_sha256'],'Snapshot source/log identity differs')
    need((folder/'SNAPSHOT_README.md').read_bytes()==NOTICE.encode('utf-8'),'Notice differs')
    need({p.relative_to(folder).as_posix() for p in folder.rglob('*') if p.is_file()}==expected,'Extra/missing snapshot files')
    return {'status':'verified','snapshot_sha256':sha(folder/'snapshot_manifest.json'),'copied_files':len(c),
            'external_dependencies':len(e),'checked_stages':state['checked_stages'],'models_or_arrays_loaded':False,'large_model_original_bytes_verified':BIG_MODEL in e,'large_model_copied':False}


def create():
    need(not DEST.exists(),'Snapshot exists; never overwrite')
    need(not list(DEST.parent.glob('_round32_experiments_staging_*')),'Previous staging must be preserved and inspected')
    copied,external,state=collect();stage=DEST.parent/('_round32_experiments_staging_'+uuid.uuid4().hex)
    stage.mkdir(parents=True,exist_ok=False)
    for name,item in copied.items():
        target=safe('files/'+name,stage,False);need(not target.exists(),'Copy destination exists')
        target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(safe(name),target)
        need(sha(target)==sha(safe(name))==item['sha256'] and target.stat().st_size==item['bytes'],'Copy/source changed')
    (stage/'SNAPSHOT_README.md').write_bytes(NOTICE.encode('utf-8'))
    m={'kind':'round32_experiment_snapshot','created_at_utc':datetime.now(timezone.utc).isoformat(),'state':state,
       'copied_files':copied,'external_dependencies':external,'copied_file_count':len(copied),
       'external_dependency_count':len(external),'copied_bytes':sum(v['bytes'] for v in copied.values()),'self_contained':False}
    path=stage/'snapshot_manifest.json'
    with path.open('xb') as f:f.write(encoded(m))
    with path.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(path)+'\n')
    need(context()==state['mutable_context_at_creation'],'Mutable context changed during copy')
    need(not DEST.exists(),'Concurrent destination appeared; preserve staging');stage.rename(DEST)
    print(json.dumps({'status':'created','snapshot_sha256':sha(DEST/'snapshot_manifest.json'),
          'copied_files':len(copied),'copied_bytes':m['copied_bytes'],'external_dependencies':len(external),
          'checked_stages':state['checked_stages'],'fresh_verify_required':True,'large_model_copied':False},ensure_ascii=False),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);mode=parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--create',action='store_true');mode.add_argument('--verify',action='store_true')
    args=parser.parse_args()
    if args.verify:print(json.dumps(verify_at(DEST),ensure_ascii=False),flush=True)
    else:create()
