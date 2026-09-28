from pathlib import Path
from datetime import datetime, timezone
import hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[3]
OUT=Path(__file__).resolve().parent
SOURCE=ROOT/"src/round90_offset_data.py"
EXPECTED="4f58575bff75b69fd900af84f9152dc3e900f34ad6909e0f2e0366bb67c928f1"
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(name,d):
 p=OUT/name
 with p.open("x",encoding="utf-8",newline="\n") as f:f.write(json.dumps(d,ensure_ascii=False,indent=2,allow_nan=False)+"\n")
 with p.with_suffix(".sha256").open("x",encoding="ascii") as f:f.write(sha(p)+"\n")
assert sha(SOURCE)==EXPECTED and not (ROOT/"outputs/round90/preparation").exists()
command=[str(ROOT/".venv/Scripts/python.exe"),"-u","-B","-X","utf8",str(SOURCE),"--prepare"]
start=datetime.now(timezone.utc).isoformat();tick=time.monotonic()
write("started.json",{"command":command,"started_at_utc":start,"source_sha256":EXPECTED,
 "authorized_scope":"one target-free preparation; zero model fits, price tables, test or scoring",
 "selfcheck_sha256":sha(ROOT/"reports/round90_data_selfcheck.json"),"automatic_retry":False})
code=1;failure=None
try:
 with (OUT/"prepare.log").open("x",encoding="utf-8") as f:
  done=subprocess.run(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=False)
  code=done.returncode
except Exception as exc:
 failure=repr(exc)
result={"started_at_utc":start,"completed_at_utc":datetime.now(timezone.utc).isoformat(),
 "elapsed_seconds":time.monotonic()-tick,"exit_code":code,"failure":failure,
 "source_sha256_before":EXPECTED,"source_sha256_after":sha(SOURCE),"source_unchanged":sha(SOURCE)==EXPECTED,
 "automatic_retry":False,"log_sha256":sha(OUT/"prepare.log") if (OUT/"prepare.log").exists() else None}
write("execution.json",result)
print(json.dumps(result,ensure_ascii=False),flush=True)
raise SystemExit(code)
