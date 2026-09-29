"""Run one frozen R103 diagnostic with preserved stdout/stderr and terminal status."""
from pathlib import Path
from datetime import datetime, timezone
import subprocess, sys, time, json, hashlib
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def main():
 request=HERE/'request.json'
 assert request.is_file()
 assert not (ROOT/'outputs/round103').exists()
 command=[sys.executable,'-B','-X','utf8',str(ROOT/'src/round103_complementarity_diagnostics.py'),'--run','--request',str(request),'--out',str(ROOT/'outputs/round103')]
 start=datetime.now(timezone.utc).isoformat(); tick=time.monotonic()
 with (HERE/'stdout.log').open('x',encoding='utf8') as out, (HERE/'stderr.log').open('x',encoding='utf8') as err:
  result=subprocess.run(command,cwd=ROOT,stdout=out,stderr=err)
 receipt={'started_at_utc':start,'completed_at_utc':datetime.now(timezone.utc).isoformat(),'exit_code':result.returncode,'elapsed_seconds':time.monotonic()-tick,'command':command,'request_sha256':hashlib.sha256(request.read_bytes()).hexdigest(),'new_model_fits':0,'automatic_retry':False}
 with (HERE/'execution.json').open('x',encoding='utf8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2);f.write('\n')
 print(json.dumps(receipt,ensure_ascii=False),flush=True)
 if result.returncode:raise SystemExit(result.returncode)
if __name__=='__main__':main()
