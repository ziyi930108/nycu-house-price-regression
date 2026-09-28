"""Run one frozen R100 phase; preserve its log, timing and failure state."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,subprocess,sys,time
ROOT=Path(__file__).resolve().parents[2]
HERE=Path(__file__).resolve().parent

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def now():return datetime.now(timezone.utc).isoformat()
def write(p,r):
    with p.open('x',encoding='utf-8') as f:json.dump(r,f,ensure_ascii=False,indent=2);f.write('\n')
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['train','evaluate'],required=True);ap.add_argument('--request-sha',required=True);a=ap.parse_args()
    request=HERE/('research_request.json' if a.phase=='train' else 'evaluation_request.json')
    assert sha(request)==a.request_sha,'Frozen request changed'
    source=ROOT/'src'/('round100_temporal_prior_experiment.py' if a.phase=='train' else 'round100_temporal_prior_evaluate.py')
    args=['--run','--request',str(request)] if a.phase=='train' else ['--sealed-manifest',str(ROOT/'outputs/round100/sealed_manifest.json'),'--evaluation-request',str(request),'--out',str(ROOT/'outputs/round100/evaluation')]
    cmd=[sys.executable,'-u','-B','-X','utf8',str(source),*args]
    started=HERE/(a.phase+'_outer_started.json');completed=HERE/(a.phase+'_outer_execution.json');log=HERE/(a.phase+'.log')
    assert not any(p.exists() for p in (started,completed,log)),'Existing phase evidence; no retries or overwrite'
    record={'started_at_utc':now(),'phase':a.phase,'command':cmd,'request_sha256':a.request_sha,'source_sha256':sha(source),'automatic_retry':False}
    write(started,record);tick=time.monotonic()
    with log.open('x',encoding='utf-8') as f:
        child=subprocess.Popen(cmd,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT)
        print(json.dumps({'phase':a.phase,'pid':child.pid,'log':str(log)}),flush=True)
        code=child.wait()
    result={**record,'exit_code':code,'completed_at_utc':now(),'elapsed_seconds':time.monotonic()-tick,'request_unchanged':sha(request)==a.request_sha,'source_unchanged':sha(source)==record['source_sha256']}
    write(completed,result);print(json.dumps(result,ensure_ascii=False),flush=True)
    return code
if __name__=='__main__':raise SystemExit(main())
