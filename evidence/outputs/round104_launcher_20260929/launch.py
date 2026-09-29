import sys,time,subprocess,traceback,json
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c
import round104_experiment as driver
import round104_evaluate as evaluator
folder=Path(__file__).resolve().parent
request=folder/'research_request.json'; evaluation=folder/'evaluation_request.json'
py=str(c.ROOT/'.venv/Scripts/python.exe')
start=time.monotonic();failure=None;steps=[]
c.write(folder/'started.json',{'started_at_utc':c.now(),'source':c.record(__file__),'query_scores_computed':False})
try:
 teacher='outputs/round104_teacher/models/sealed_manifest.json'
 t=time.monotonic();driver.freeze_request(request,teacher)
 sources={}
 for collection in (driver.source_hashes(),evaluator.source_hashes(),{p:c.sha(p) for p in evaluator.TRAINER_SOURCES}):
  for p,h in collection.items():c.need(p not in sources or sources[p]==h,'Source conflict');sources[p]=h
 check=c.read('reports/round104_evaluator_selfcheck.json')
 c.need(check['status']=='passed' and check['source_sha256']==c.sha(evaluator.__file__),'Evaluator receipt differs')
 c.verify_hashes(check['source_hashes'])
 c.write(evaluation,{'schema_version':1,'created_at_utc':c.now(),'plan':{'path':evaluator.PLAN,'sha256':evaluator.PLAN_SHA},
  'parameters':evaluator.parameters(),'launch_request':c.record(request),'source_sha256':sources,
  'teacher_seal':c.record(teacher),'protected_sha256':c.protected_hashes(),
  'selfcheck':c.record('reports/round104_evaluator_selfcheck.json'),'student_prepare_or_fit_started':False,'query_prices_read':False})
 c.write(folder/'freeze_execution.json',{'exit_code':0,'elapsed_seconds':time.monotonic()-t,'research_request':c.record(request),'evaluation_request':c.record(evaluation),'source_count':len(sources),'query_prices_read':False})
 print('Research and evaluation sources frozen before all six student fits',flush=True)
 jobs=[('students',['src/round104_experiment.py','--run','--request',str(request)]),
       ('evaluation',['src/round104_evaluate.py','--sealed-manifest','outputs/round104/sealed_manifest.json','--evaluation-request',str(evaluation),'--out','outputs/round104/evaluation'])]
 for name,args in jobs:
  c.verify_hashes(sources);tick=time.monotonic();print('Starting '+name,flush=True)
  cmd=[py,'-u','-B','-X','utf8',*args]
  with (folder/(name+'.log')).open('x',encoding='utf8') as log:r=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT)
  item={'step':name,'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-tick,'command':cmd,'log':c.record(folder/(name+'.log'))}
  c.write(folder/(name+'_execution.json'),item);steps.append(item)
  c.need(r.returncode==0,'R104 step failed; no automatic retry: '+name)
  print('Completed '+name,flush=True)
except Exception:failure=traceback.format_exc();raise
finally:c.write(folder/'execution.json',{'exit_code':int(failure is not None),'failure':failure,'steps':steps,'elapsed_seconds':time.monotonic()-start,'finished_at_utc':c.now(),'automatic_retry':False})
