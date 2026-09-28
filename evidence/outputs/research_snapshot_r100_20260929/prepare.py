"""Prepare only exact, explicit R100 documentation/source copies; no training/publishing."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib, json

ROOT=Path(__file__).resolve().parents[2]
REPO=ROOT/"deliverables/v22_github"
OUT=REPO/"evidence/outputs/research_snapshot_r100_20260929/preparation"
BASE="c34842980f8abc4bcc15dc9d5940fe2d50ac118f"
SOURCES=[
 "src/round100_contract.py","src/round100_prepare_fit_labels.py",
 "src/round100_temporal_prior_x_audit.py","src/round100_temporal_prior.py",
 "src/round100_temporal_prior_experiment.py","src/round100_temporal_prior_evaluate.py",
 "reports/round100_plan.md","reports/round100_x_audit_plan.md","reports/round100_evaluation_contract.md",
 "outputs/round100_x_audit/report.md","outputs/round100_evaluator_prefreeze/source_attempt1.py",
 "outputs/round100_evaluator_prefreeze/source_attempt2.py","outputs/round100_evaluator_prefreeze/audit_order_revision.json",
 "outputs/round100_launcher_20260929/fit_label_source_review.json",
 "outputs/research_snapshot_r100_20260929/prepare.py"]
for stem in [
 "reports/round100_helper_selfcheck","reports/round100_driver_selfcheck",
 "reports/round100_fit_label_selfcheck","reports/round100_evaluator_selfcheck",
 "reports/round100_evaluator_selfcheck_attempt1","reports/round100_evaluator_selfcheck_attempt2",
 "outputs/round100_x_audit/results","outputs/round100_x_audit/execution",
 "outputs/round100_x_audit_preflight/request","outputs/round100_x_audit_preflight/selfcheck",
 "outputs/round100_inputs/manifest","outputs/round100_inputs/started","outputs/round100_inputs/execution"]:
 SOURCES += [stem+".json",stem+".sha256"]

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open("x",encoding="utf8",newline="\n") as f:
  json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n")

def main():
 assert not OUT.exists(),"Do not overwrite prior preparation"
 assert len(set(SOURCES))==len(SOURCES)
 for name in SOURCES:
  p=ROOT/name
  assert p.is_file(),name
  assert p.suffix in {".py",".md",".json",".sha256"},name
  assert not any(x in p.parts for x in ("private","prefix_states")),name
 # A metadata manifest may identify private arrays, but never contains their values.
 for name in SOURCES:
  if name.endswith(".json"):
   obj=json.loads((ROOT/name).read_text(encoding="utf-8-sig"))
   def inspect(value):
    if isinstance(value,dict):
     for key,item in value.items():
      assert key not in {"global_median","reference_global_median","full_fit_global_median","cells"},(name,key)
      inspect(item)
    elif isinstance(value,list):
     assert not (len(value)>100 and all(isinstance(x,(int,float)) for x in value)),name
     for item in value:inspect(item)
   inspect(obj)
 history=REPO/"research/source_manifest.json"
 historical=json.loads(history.read_text(encoding="utf8"));assert len(historical)==379
 existing={p.relative_to(REPO).as_posix():sha(p) for p in REPO.rglob("*") if p.is_file() and ".git" not in p.relative_to(REPO).parts}
 outer=json.loads((REPO/"SOURCE_MANIFEST.json").read_text(encoding="utf8"))
 assert len(outer["files"])==1590
 baseline={"created_at_utc":datetime.now(timezone.utc).isoformat(),"base_commit":BASE,
  "existing_files":existing,"published_source_count":1590,"published_plus_manifest_count":1591,
  "historical_manifest_sha256":sha(history),"outer_source_manifest_sha256":sha(REPO/"SOURCE_MANIFEST.json")}
 OUT.mkdir(parents=True)
 write(OUT/"baseline.json",baseline)
 before=OUT/"navigation_before";before.mkdir()
 for name in ("README.md","HANDOFF.md","research/README.md"):
  dst=before/name;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((REPO/name).read_bytes())
 records=[]
 for name in sorted(SOURCES):
  p=ROOT/name;dst=REPO/(("research/"+name) if name.startswith("src/") else ("evidence/"+name))
  assert not dst.exists(),str(dst)
  dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes(p.read_bytes())
  assert dst.read_bytes()==p.read_bytes()
  records.append({"source":name,"destination":dst.relative_to(REPO).as_posix(),"bytes":dst.stat().st_size,"sha256":sha(dst)})
 copy={"status":"preparation_only_no_completed_model_results_included","files":records,"copied_count":len(records),
  "copied_bytes":sum(r["bytes"] for r in records),"source_bytes_preserved":True,
  "excluded":["labels/arrays","model weights","NPZ/NPY/CSV/XLSX/pickle","complete prior prefix states/cell prices",
              "active training artifacts","private/runs contents","navigation edits","package/commit/push/ZIP"]}
 write(OUT/"copy_manifest.json",copy)
 assert all(sha(REPO/n)==digest for n,digest in existing.items()),"Existing repo files changed"
 assert sha(history)==baseline["historical_manifest_sha256"]
 result={"status":"passed","existing_files_unchanged":len(existing),"copied_count":len(records),
  "copied_bytes":copy["copied_bytes"],"all_source_bytes_exact":True,"navigation_not_edited":True,
  "historical_379_manifest_and_files_protected_by_full_baseline":True,
  "formal_REPORT_PDF_PPTX_protected_by_full_baseline":True,"real_fit_score_calls":0,
  "model_completion_claimed":False,"package_commit_push_zip_executed":False,
  "copy_manifest_sha256":sha(OUT/"copy_manifest.json")}
 write(OUT/"preparation_verification.json",result)
 print(json.dumps(result,ensure_ascii=False))
if __name__=="__main__":main()
