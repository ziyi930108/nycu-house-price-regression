"""Archive completed R10 feasibility and the separately labelled v3 diagnostic.
No new model or submission is created; original records are never overwritten.
"""
from __future__ import annotations
import argparse, hashlib, json, shutil
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'runs/round10_research'
DIAG=ROOT/'outputs/diagnostics/v3_cluster_stability'
def need(ok,message):
 if not ok:raise RuntimeError(message)
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def rel(path):return Path(path).resolve().relative_to(ROOT).as_posix()
def safe(name):
 p=(ROOT/name).resolve();need(p.is_relative_to(ROOT) and p.is_file(),'Missing/unsafe input: '+str(name));return p
def locked(path):
 path=Path(path);need(path.with_suffix('.sha256').read_text(encoding='ascii').strip()==sha(path),'Receipt differs: '+str(path));return read(path)
def encoded(v):return (json.dumps(v,ensure_ascii=False,sort_keys=True,indent=2,allow_nan=False)+'\n').encode('utf-8')
def collect():
 inventory=locked(ROOT/'outputs/round10/research/inventory.json')
 decision=locked(ROOT/'outputs/round10/research/research_conclusion.json')
 review=read(ROOT/'outputs/round10/research/independent_review.json')
 plan=locked(DIAG/'plan.json');diagnostic=locked(DIAG/'results.json')
 need(decision['status']=='not_advanced_to_model' and not decision['formal_candidate_selected'],'Wrong research decision')
 need(all(decision[k]==0 for k in ['new_model_count','new_mae_count','new_submission_count']),'Unexpected R10 model result')
 need(inventory['model_training_or_scoring'] is False and inventory['formal_candidate_selected'] is False,'Research performed ML scoring')
 need(review['decision_verified']['status']=='not_advanced_to_model','Independent research decision differs')
 need(diagnostic['plan_sha256']==sha(DIAG/'plan.json') and diagnostic['selection_changed'] is False and diagnostic['recommended_version']=='v3','Wrong diagnostic identity')
 need(sha(DIAG/'results.json')=='f1106aa8d76719579740d55da834bcf901258dc87dbb9280cd963542aafc69b6','Diagnostic result differs')
 need(read(ROOT/'runs/versions.json')['recommended_version']=='v3','Registry recommendation differs')
 need((DIAG/'code_review.json').is_file() and (DIAG/'result_review.json').is_file(),'Independent diagnostic reviews missing')
 paths=[Path(__file__).resolve(),ROOT/'src/round10_remarks_inventory.py',ROOT/'src/v3_cluster_stability.py']
 paths+=list((ROOT/'reports').glob('round10*.md'))
 paths += [ROOT/'reports/v3_cluster_stability_plan.md',ROOT/'reports/v3_cluster_stability_report.md']
 paths += [p for folder in [ROOT/'outputs/round10/research',DIAG] for p in folder.rglob('*') if p.is_file()]
 need(all(p.is_file() and p.resolve().is_relative_to(ROOT) for p in paths),'Unsafe research artifact')
 need(not (ROOT/'outputs/round10/models').exists() and not (ROOT/'outputs/round10/evaluation').exists(),'Unexpected R10 model/evaluation directory')
 copied={rel(p):sha(p) for p in sorted(set(paths))};dependencies={}
 def bind(name,digest):
  p=safe(name);name=rel(p)
  need(name not in dependencies or dependencies[name]==digest,'Conflicting dependency SHA: '+name)
  need(sha(p)==digest,'Recorded dependency changed: '+name);dependencies[name]=digest
 bind(inventory['source_workbook'],inventory['source_workbook_sha256'])
 for mapping in [inventory['source_code_sha256'],review['verified_file_sha256'],plan['dependency_sha256'],diagnostic['artifact_sha256']]:
  for name,digest in mapping.items():bind(name,digest)
 for name,digest in copied.items():
  if name in dependencies:need(dependencies[name]==digest,'Copy/input dependency conflict')
 dependencies={k:v for k,v in dependencies.items() if k not in copied}
 return copied,dependencies,{'research_decision':'not_advanced_to_model','research_used_targets':False,
  'diagnostic_used_observed_targets_for_fixed_prediction_errors':True,'new_models':0,'new_submission_csvs':0,
  'research_new_candidate_mae_values':0,'diagnostic_reuses_original_v2_v3_point_scores':True,'recommended_version':'v3',
  'research_inventory_sha256':sha(ROOT/'outputs/round10/research/inventory.json'),
  'research_decision_sha256':sha(ROOT/'outputs/round10/research/research_conclusion.json'),
  'diagnostic_plan_sha256':sha(DIAG/'plan.json'),'diagnostic_results_sha256':sha(DIAG/'results.json')}

def create():
 need(not DEST.exists(),'Snapshot exists; use --verify, never overwrite')
 copied,external,state=collect();DEST.mkdir(parents=True)
 records={}
 for name,digest in copied.items():
  source=safe(name);target=(DEST/'files'/name).resolve()
  need(target.is_relative_to(DEST.resolve()),'Unsafe snapshot destination')
  target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(source,target)
  need(sha(target)==digest and sha(source)==digest,'Copy differs: '+name)
  records[name]={'backup_path':rel(target),'sha256':digest,'bytes':target.stat().st_size}
 for name,digest in external.items():need(sha(safe(name))==digest,'External input changed during copy')
 manifest={'kind':'round10_research_with_separate_frozen_v3_stability_diagnostic','created_at_utc':datetime.now(timezone.utc).isoformat(),
  'state':state,'copied_files':records,'external_dependencies':external,'copied_file_count':len(records),
  'copied_bytes':sum(v['bytes'] for v in records.values()),'external_dependency_count':len(external),
  'old_sources_and_outputs_modified':False,'registry_modified':False,
  'notes':'Research is X-only; the separate diagnostic reads original train labels only to assess already-frozen predictions. Neither creates a new model or submission.'}
 p=DEST/'snapshot_manifest.json'
 with p.open('xb') as f:f.write(encoded(manifest))
 with p.with_suffix('.sha256').open('x',encoding='ascii') as f:f.write(sha(p)+'\n')
 verify()
 print(json.dumps({'status':'created','snapshot_sha256':sha(p),'files':len(records),'bytes':manifest['copied_bytes'],'external_dependencies':len(external)}),flush=True)

def verify():
 p=DEST/'snapshot_manifest.json';manifest=locked(p)
 need(manifest['state']['recommended_version']=='v3' and manifest['state']['research_decision']=='not_advanced_to_model','Wrong snapshot decision')
 for name,record in manifest['copied_files'].items():
  backup=safe(record['backup_path']);need(backup.is_relative_to(DEST.resolve()),'Backup escaped snapshot')
  need(backup.stat().st_size==record['bytes'] and sha(backup)==record['sha256'],'Snapshot copy changed: '+name)
  need(sha(safe(name))==record['sha256'],'Frozen original changed: '+name)
 for name,digest in manifest['external_dependencies'].items():need(sha(safe(name))==digest,'Recorded external input changed: '+name)
 print(json.dumps({'status':'verified','snapshot_sha256':sha(p),'files':manifest['copied_file_count'],'external_dependencies':manifest['external_dependency_count']}),flush=True)

if __name__=='__main__':
 parser=argparse.ArgumentParser();parser.add_argument('--verify',action='store_true');args=parser.parse_args()
 verify() if args.verify else create()
