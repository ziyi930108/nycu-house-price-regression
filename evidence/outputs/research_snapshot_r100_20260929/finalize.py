"""R100 final exact-copy documentation snapshot. No ML, package, git or publish."""
from pathlib import Path
import json,hashlib,re,urllib.parse
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/"deliverables/v22_github"
SNAP=REPO/"evidence/outputs/research_snapshot_r100_20260929"
FINAL=SNAP/"final"
STAGES=("development","confirmation","auxiliary")
NAV=("README.md","HANDOFF.md","research/README.md","docs/RESEARCH_UPDATES_20260929_TEMPORAL_PRIOR.md")
SOURCES=[
 "src/round100_temporal_prior_evaluate_v2.py","src/round100_independent_review.py","src/summarize_round100.py",
 "reports/round100_driver_preflight.md","reports/round100_helper_source_review.md","reports/round100_helper_source_review.json",
 "reports/round100_evaluation_contract_v2.md","reports/round100_evaluator_selfcheck_v2.json","reports/round100_evaluator_selfcheck_v2.sha256",
 "reports/round100_summary.md","reports/round100_independent_review.md","reports/round100_next_direction_inventory.md",
 "outputs/round100/evaluation/report.md","outputs/round100_independent_review/results.json","outputs/round100_independent_review/results.sha256",
 "outputs/round100_independent_review/execution.json","outputs/round100_independent_review/execution.sha256",
 "outputs/round100_audit_prefreeze/source_attempt1.py","outputs/round100_audit_prefreeze/source_attempt1.json",
 "outputs/round100_evaluator_v2_revision/actual_metadata_check.json",
 "outputs/round100_evaluator_v2_revision/engineering_diff.patch",
 "outputs/round100_evaluator_v2_revision/final_receipt.json",
 "outputs/round100_evaluator_v2_revision/original_v1_source.py",
 "outputs/round100_evaluator_v2_revision/revision.json",
 "outputs/round100_launcher_20260929/freeze_requests.py",
 "outputs/round100_launcher_20260929/launch.py",
 "outputs/round100_launcher_20260929/launch_evaluate_v2.py",
 "outputs/round100_launcher_20260929/evaluate_outer_execution.json",
 "outputs/round100_launcher_20260929/evaluate_outer_started.json",
 "outputs/round100_launcher_20260929/train_outer_execution.json",
 "outputs/round100_launcher_20260929/train_outer_started.json",
 "outputs/research_snapshot_r100_20260929/finalize.py"]
for stem in ["outputs/round100/started","outputs/round100/sealed_manifest","outputs/round100/execution",
 "outputs/round100/evaluation/results","outputs/round100/evaluation/prelabel_gate","outputs/round100/evaluation/execution",
 "outputs/round100_launcher_20260929/evaluation_request","outputs/round100_launcher_20260929/evaluation_request_v2",
 "outputs/round100_launcher_20260929/research_request","outputs/round100_launcher_20260929/freeze_verification",
 "outputs/round100_summary/summary","outputs/round100_summary/receipt","outputs/round100_summary/execution"]:
 SOURCES += [stem+".json",stem+".sha256"]
for s in STAGES:
 for stem in ["stage_seal","preparation","execution"]:
  SOURCES += [f"outputs/round100/{s}/{stem}.json",f"outputs/round100/{s}/{stem}.sha256"]
 SOURCES += [f"outputs/round100/evaluation/{s}/results.json",f"outputs/round100/evaluation/{s}/results.sha256"]
 for arm in ("C","T"):
  for stem in ("fit_started","execution"):
   SOURCES += [f"outputs/round100/{s}/{arm}/{stem}.json",f"outputs/round100/{s}/{arm}/{stem}.sha256"]
  SOURCES += [f"outputs/round100/{s}/{arm}/model/run.json",f"outputs/round100/{s}/{arm}/model/training.json"]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf8"))
def write(p,v):
 with p.open("x",encoding="utf8",newline="\n") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n")

def inspect(value,name):
 if isinstance(value,dict):
  for k,v in value.items():
   assert k not in {"global_median","reference_global_median","full_fit_global_median","cells"},(name,k)
   inspect(v,name)
 elif isinstance(value,list):
  assert not (len(value)>100 and all(isinstance(x,(int,float)) for x in value)),name
  for v in value:inspect(v,name)

def main():
 assert not FINAL.exists(),"Preserve final receipt"
 baseline=read(SNAP/"preparation/baseline.json")
 preparation=read(SNAP/"preparation/copy_manifest.json")
 assert all(sha(REPO/name)==digest for name,digest in baseline["existing_files"].items()),"Baseline changed before final"
 # Require all completion evidence before any navigation edit.
 for name in SOURCES:assert (ROOT/name).is_file(),name
 assert len(set(SOURCES))==len(SOURCES)
 for name in ["outputs/round100/execution.json","outputs/round100/evaluation/execution.json","outputs/round100_independent_review/execution.json"]:
  assert read(ROOT/name)["exit_code"]==0,name
 for name in SOURCES:
  p=ROOT/name
  assert p.suffix in {".py",".md",".json",".sha256",".patch"}
  assert "prefix_states" not in p.parts and p.name!="temporal_prior_evidence.json"
  if p.suffix==".json":inspect(read(p),name)
 FINAL.mkdir()
 before=FINAL/"navigation_before";before.mkdir()
 for name in NAV:
  p=REPO/name
  if p.exists():
   q=before/name;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes(p.read_bytes())
 copied=[]
 for name in sorted(SOURCES):
  source=ROOT/name;destination=REPO/(("research/"+name) if name.startswith("src/") else ("evidence/"+name))
  assert not destination.exists(),destination
  destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(source.read_bytes())
  assert source.read_bytes()==destination.read_bytes()
  copied.append({"source":name,"destination":destination.relative_to(REPO).as_posix(),"bytes":source.stat().st_size,"sha256":sha(source)})
 copy={"status":"completed_source_metadata_copy","files":copied,"copied_count":len(copied),
       "copied_bytes":sum(r["bytes"] for r in copied),"preparation_count":preparation["copied_count"],
       "total_original_copies":len(copied)+preparation["copied_count"],
       "total_copied_bytes":sum(r["bytes"] for r in copied)+preparation["copied_bytes"],
       "source_bytes_preserved":True,"excluded":["705 complete prefix states","per-prefix price-bearing temporal_prior_evidence",
          "labels","NPY/NPZ/CSV/XLSX/pickle","model.pt/state","private/runs contents"],
       "omitted_price_state_identity_only":"stage_seal artifact_sha256 retains their original SHA"}
 write(FINAL/"copy_manifest.json",copy)
 print(json.dumps({"copy_complete":True,**{k:copy[k] for k in ("copied_count","copied_bytes","total_original_copies","total_copied_bytes")}}))
 # Four navigation documents are generated by a separate narrow finishing step,
 # preserving this exact source/evidence copy and its immutable receipt.
if __name__=="__main__":main()
