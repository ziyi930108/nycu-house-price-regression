"""Run frozen R97 reconstruction once; independent scoring is launched separately."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,subprocess,time,traceback
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,v):
 with p.open("x",encoding="utf-8") as f:json.dump(v,f,ensure_ascii=False,indent=2);f.write("\n")
def main():
 request=json.loads((HERE/"research_request.json").read_text(encoding="utf-8"))
 assert request["approval_for_offset_rebuild"] is True
 for p,h in request["frozen_source_and_plan"].items():assert sha(ROOT/p)==h,p
 start=now();timer=time.monotonic();steps=[];failure=None;code=0
 commands=[("main",request["main_command"])]
 write(HERE/"pipeline_started.json",dict(started_at_utc=start,commands=commands,request_sha256=sha(HERE/"research_request.json"),automatic_retry=False))
 try:
  for label,cmd in commands:
   for p,h in request["frozen_source_and_plan"].items():assert sha(ROOT/p)==h,p
   before=now();tick=time.monotonic()
   with (HERE/(label+".log")).open("x",encoding="utf-8") as f:
    r=subprocess.run(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False,creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
   item=dict(step=label,started_at_utc=before,completed_at_utc=now(),elapsed_seconds=time.monotonic()-tick,exit_code=r.returncode,command=cmd,log_sha256=sha(HERE/(label+".log")))
   write(HERE/(label+"_execution.json"),item);steps.append(item)
   print(json.dumps(item),flush=True)
   if r.returncode:raise RuntimeError(label+" failed; original evidence retained")
 except Exception:
  code=1;failure=traceback.format_exc();print(failure,flush=True)
 write(HERE/"pipeline_execution.json",dict(started_at_utc=start,completed_at_utc=now(),elapsed_seconds=time.monotonic()-timer,exit_code=code,failure=failure,steps=steps,automatic_retry=False,source_sha256=sha(Path(__file__))))
 return code
if __name__=="__main__":raise SystemExit(main())
