import sys,time,subprocess,traceback,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c
import round105_native_average_evaluate as evaluator
folder=Path(__file__).resolve().parent
request=folder/'research_request.json';evaluation=folder/'evaluation_request.json'
py=str(c.ROOT/'.venv/Scripts/python.exe')
tick=time.monotonic();failure=None;steps=[]
c.write(folder/'started.json',{'started_at_utc':c.now(),'source':c.record(__file__),'new_fits':0})
def command(name,args):
 start=time.monotonic();print('Starting '+name,flush=True)
 cmd=[py,'-u','-B','-X','utf8',*args]
 with (folder/(name+'.log')).open('x',encoding='utf8') as log:r=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
 v={'step':name,'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-start,'command':cmd,'log':c.record(folder/(name+'.log'))}
 c.write(folder/(name+'_execution.json'),v);steps.append(v)
 c.need(r.returncode==0,'R105 step failed; no automatic retry: '+name)
 print('Completed '+name,flush=True)
try:
 command('freeze',['src/round105_native_average.py','--freeze-request',str(request)])
 launch=c.read(request);sources=launch['source_sha256'].copy()
 for p,h in evaluator.source_hashes().items():c.need(p not in sources or sources[p]==h,'Source conflict');sources[p]=h
 c.verify_hashes(sources)
 check=c.read('reports/round105_evaluator_selfcheck.json')
 c.need(check['status']=='passed','Evaluator selfcheck incomplete')
 c.need(check['plan_sha256']==evaluator.PLAN_SHA and check['source_hashes']==evaluator.source_hashes(),'Evaluator selfcheck stale')
 metadata=c.read('reports/round105_evaluator_metadata_preflight.json')
 c.need(metadata['status']=='passed' and metadata['plan_sha256']==evaluator.PLAN_SHA and metadata['source_hashes']==evaluator.source_hashes(),'Evaluator metadata stale')
 c.need(metadata['driver_source_sha256']==c.sha('src/round105_native_average.py'),'Driver metadata stale')
 c.write(evaluation,{'schema_version':1,'created_at_utc':c.now(),'plan':{'path':evaluator.PLAN,'sha256':evaluator.PLAN_SHA},
   'parameters':evaluator.parameters(),'source_sha256':sources,'launch_request':c.record(request),
   'protected_sha256':c.protected_hashes(),'selfcheck':c.record('reports/round105_evaluator_selfcheck.json'),
   'candidate_generation_started':False,'query_prices_read':False})
 print('Research and evaluator frozen before candidate prediction',flush=True)
 command('predictions',['src/round105_native_average.py','--run','--request',str(request)])
 c.verify_hashes(sources)
 command('evaluation',['src/round105_native_average_evaluate.py','--sealed-manifest','outputs/round105/sealed_manifest.json',
    '--evaluation-request',str(evaluation),'--out','outputs/round105/evaluation'])
except Exception:failure=traceback.format_exc();raise
finally:c.write(folder/'execution.json',{'exit_code':int(failure is not None),'failure':failure,'steps':steps,
   'elapsed_seconds':time.monotonic()-tick,'finished_at_utc':c.now(),'automatic_retry':False,'new_fits':0})
