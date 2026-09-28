"""Launch the separately frozen evaluator only after the full R97 main succeeds."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_text(encoding="utf-8"))
def main():
 assert read(ROOT/"outputs/round97/execution.json")["exit_code"]==0
 assert read(HERE/"pipeline_execution.json")["exit_code"]==0
 request=read(HERE/"evaluation_request.json")
 for name,digest in request["source_sha256"].items():assert sha(ROOT/name)==digest,name
 assert (HERE/"evaluation_request.sha256").read_text().strip()==sha(HERE/"evaluation_request.json")
 cmd=[str(ROOT/".venv/Scripts/python.exe"),"-B","-X","utf8","src/round97_offset_evaluate.py",
  "--sealed-manifest","outputs/round97/sealed_manifest.json",
  "--evaluation-request","outputs/round97_launcher_20260929/evaluation_request.json",
  "--out","outputs/round97/evaluation"]
 started=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
 with (HERE/"evaluation.log").open("x",encoding="utf-8") as f:
  done=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
 receipt=dict(started_at_utc=started,completed_at_utc=datetime.now(timezone.utc).isoformat(),
  elapsed_seconds=time.monotonic()-tick,exit_code=done.returncode,command=cmd,automatic_retry=False,
  source_sha256=sha(Path(__file__)),request_sha256=sha(HERE/"evaluation_request.json"),
  log_sha256=sha(HERE/"evaluation.log"))
 with (HERE/"evaluation_execution.json").open("x",encoding="utf-8") as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write("\n")
 print(json.dumps(receipt))
 return done.returncode
if __name__=="__main__":raise SystemExit(main())

