"""Run the frozen R94 experiment then saved-prediction diagnostics, once."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent
PY=ROOT/".venv/Scripts/python.exe"
def now():return datetime.now(timezone.utc).isoformat()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,d):
 with p.open("x",encoding="utf8") as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write("\n")
def main():
 request=json.loads((HERE/"research_request.json").read_text(encoding="utf8"))
 assert request["approval_for_three_fits"] is True and request["resource_gate_passed"] is True
 for n,h in request["frozen_source_and_plan"].items():assert sha(ROOT/n)==h,n
 commands=[
 ("main",request["main_command"]),
 ("diagnostics",[str(PY),"-B","-X","utf8","src/candidate_diagnostics_v3.py","--round","94","--candidate","R94",
                 "--extra-control","v22:79:R79","--extra-control","v24:83:R83"]),
 ("paired_v22",[str(PY),"-B","-X","utf8","src/paired_candidate_bootstrap.py","--round","94","--candidate","R94",
                "--control-round","79","--control-name","v22","--control-key","R79"]),
 ("paired_v24",[str(PY),"-B","-X","utf8","src/paired_candidate_bootstrap.py","--round","94","--candidate","R94",
                "--control-round","83","--control-name","v24","--control-key","R83"]),
 ("review",[str(PY),"-B","-X","utf8","src/round94_review.py"])]
 started=now();tick=time.monotonic();steps=[];error=None;code=0
 write(HERE/"pipeline_started.json",{"started_at_utc":started,"commands":commands,"request_sha256":sha(HERE/"research_request.json"),"source_sha256":sha(Path(__file__)),"automatic_retry":False})
 try:
  for name,command in commands:
   begin=now();timer=time.monotonic()
   with (HERE/(name+".log")).open("x",encoding="utf8") as f:
    done=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
   event={"step":name,"started_at_utc":begin,"completed_at_utc":now(),"elapsed_seconds":time.monotonic()-timer,
          "command":command,"exit_code":done.returncode,"log_sha256":sha(HERE/(name+".log"))}
   write(HERE/(name+"_execution.json"),event);steps.append(event)
   print(json.dumps({"step":name,"exit_code":done.returncode,"elapsed_seconds":event["elapsed_seconds"]}),flush=True)
   if done.returncode:raise RuntimeError(name+" failed; existing outputs are retained")
 except Exception:
  code=1;error=traceback.format_exc();print(error,flush=True)
 write(HERE/"pipeline_execution.json",{"started_at_utc":started,"completed_at_utc":now(),"elapsed_seconds":time.monotonic()-tick,
  "exit_code":code,"failure":error,"steps":steps,"automatic_retry":False,"source_sha256":sha(Path(__file__))})
 return code
if __name__=="__main__":raise SystemExit(main())
