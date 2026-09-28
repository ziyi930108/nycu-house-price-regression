"""Serial isolated R101 diagnostics; this parent reads metadata only."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
import time
import round100_contract as c


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--request',required=True)
    ap.add_argument('--request-sha',required=True)
    a=ap.parse_args()
    c.need(c.sha(a.request)==a.request_sha,'Frozen request changed')
    request=c.read(a.request)
    c.need(request['stage_order']==list(c.STAGES),'Stage order changed')
    c.verify_hashes(request['source_sha256'])
    c.need(request['source_sha256'][c.rel(__file__)]==c.sha(__file__),'Unfrozen launcher')
    out=c.ROOT/'outputs/round101'
    c.need(not out.exists(),'Preserve existing diagnostic run')
    out.mkdir()
    started={'started_at_utc':c.now(),'request':c.record(a.request),'source_sha256':request['source_sha256'],
             'automatic_retry':False,'stage_order':list(c.STAGES)}
    c.write(out/'started.json',started)
    tick=time.monotonic();failure=None;stages={}
    try:
        for stage in c.STAGES:
            command=[sys.executable,'-u','-B','-X','utf8',str(c.ROOT/'src/round101_prior_diagnostics.py'),
                     '--request',str(c.safe(a.request)),'--stage',stage]
            with (out/(stage+'.log')).open('x',encoding='utf8') as log:
                child=subprocess.Popen(command,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
                print(json.dumps({'stage':stage,'pid':child.pid}),flush=True)
                code=child.wait()
            c.need(code==0,'Stage failed without automatic retry: '+stage)
            result=c.read(out/stage/'results.json');execution=c.read(out/stage/'execution.json')
            c.need(execution['exit_code']==0 and result['status']=='descriptive_diagnostics_completed','Incomplete stage')
            stages[stage]={'results':c.record(out/stage/'results.json'),'execution':c.record(out/stage/'execution.json'),
                           'elapsed_seconds':execution['elapsed_seconds'],
                           'eligible_rows':result['eligible_rows'],'query_rows':result['query_rows']}
        c.verify_hashes(request['source_sha256'])
        c.need(request['protected_sha256']==c.protected_hashes(),'Protected files changed')
        c.need(c.sha(a.request)==a.request_sha,'Request changed during run')
        c.write(out/'results.json',{'status':'three_isolated_diagnostics_completed','round':101,
                  'request':c.record(a.request),'stages':stages,'stage_order':list(c.STAGES),
                  'new_native_fits':0,'query_labels_read':False,'MAE_calls':0,'new_CSV':False,
                  'source_sha256':request['source_sha256'],'protected_sha256':c.protected_hashes()})
    except Exception as e:
        failure={'type':type(e).__name__,'message':str(e)}
        raise
    finally:
        result={'exit_code':int(failure is not None),'failure':failure,'elapsed_seconds':time.monotonic()-tick,
                'completed_at_utc':c.now(),'completed_stages':list(stages),'automatic_retry':False}
        c.write(out/'execution.json',result)
        print(json.dumps(result),flush=True)


if __name__=='__main__':
    main()
