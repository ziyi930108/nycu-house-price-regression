"""Freeze actual R97 launch closure and perform read-only fit-input contract checks."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,sys,time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/"src"))
import round97_offset_experiment as d
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,v):
 with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n")
def main():
 start=time.monotonic()
 assert not d.OUT.exists(),"Existing R97 output"
 mf=d.read_json(d.PREPARATION/"manifest.json")
 modules=d.load_modules()
 closure=d.source_hashes(modules,mf)
 assert sha(d.DOCUMENT)=="8372960e832bd66203374c942e0f7988e66522aedf358c894d892e82669a3a08"
 closure[d.rel(d.DOCUMENT)]=sha(d.DOCUMENT)
 closure[d.rel(HERE/"run_pipeline.py")]=sha(HERE/"run_pipeline.py")
 closure[d.rel(Path(__file__))]=sha(Path(__file__))
 checked=d.self_check()
 assert checked["status"]=="passed"
 data=modules[0].load_prepared(d.PREPARATION,d.PREPARATION_SHA)
 y,labels=d.load_fit_labels(data,sha(d.LABEL_MANIFEST))
 modules[1].log.fit_weights(data.Xfit)
 assert len(y)==272117 and len(data.Xquery)==107169
 assert labels["query_labels_exported"] is False
 checks=dict(status="passed",source_sha256=sha(ROOT/"src/round97_offset_experiment.py"),
   source_files=len(closure),fit_rows=len(y),query_rows=len(data.Xquery),
   synthetic_checks=checked["checks"],actual_fit_label_contract_loaded=True,
   pure_X_reload_validated=True,raw_pickle_deserialized=False,query_labels_read=False,
   native_fits=0,log_date_contract_passed=True,elapsed_seconds=time.monotonic()-start)
 write(HERE/"check_launch.json",checks)
 request=dict(approval_for_offset_rebuild=True,authorization="User authorized continued regression improvement; frozen historical-window validation",
  plan_sha256=sha(d.DOCUMENT),prepared_manifest_sha256=d.PREPARATION_SHA,
  fit_label_manifest_sha256=sha(d.LABEL_MANIFEST),native_fits=dict(trees=5,NN=2),
  formulas=d.FORMULAS,frozen_source_and_plan=closure,
  created_at_utc=datetime.now(timezone.utc).isoformat(),
  main_command=[str(ROOT/".venv/Scripts/python.exe"),"-u","-B","-X","utf8","src/round97_offset_experiment.py","--run","--plan-sha",sha(d.DOCUMENT)],
  evaluation_separately_frozen_before_query_labels=True,evaluation_contract_already_in_plan=True,
  automatic_retry=False,selected_submission="v22",Private_score=None)
 write(HERE/"research_request.json",request)
 print(json.dumps(checks))
if __name__=="__main__":main()
