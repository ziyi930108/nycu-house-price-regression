import sys,time,subprocess,traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c
folder=Path(__file__).resolve().parent
prep_request=folder/'preparation_request.json'
fit_request=folder/'fit_request.json'
prep='outputs/round104_teacher/preparation/manifest.json'
labels='outputs/round104_teacher/fit_labels/manifest.json'
py=str(c.ROOT/'.venv/Scripts/python.exe')
steps=[('freeze_preparation',['src/round104_teacher_preparation.py','--freeze-request',str(prep_request)]),
 ('prepare',['src/round104_teacher_preparation.py','--prepare','--request',str(prep_request),'--out','outputs/round104_teacher/preparation']),
 ('extract_fit_labels',['src/round104_teacher_preparation.py','--extract-fit-labels','--request',str(prep_request),'--preparation',prep,'--out','outputs/round104_teacher/fit_labels']),
 ('freeze_fit',['src/round104_teacher_fit.py','--freeze-request',str(fit_request),'--preparation',prep,'--labels',labels]),
 ('fit',['src/round104_teacher_fit.py','--run','--request',str(fit_request),'--out','outputs/round104_teacher/models'])]
start=time.monotonic();error=None;done=[]
c.write(folder/'started.json',{'started_at_utc':c.now(),'source':c.record(__file__),'steps':steps})
try:
 for name,args in steps:
  tick=time.monotonic();command=[py,'-u','-B','-X','utf8',*args]
  print('Starting '+name,flush=True)
  with (folder/(name+'.log')).open('x',encoding='utf8') as f:r=subprocess.run(command,cwd=c.ROOT,stdout=f,stderr=subprocess.STDOUT)
  item={'step':name,'exit_code':r.returncode,'elapsed_seconds':time.monotonic()-tick,'command':command,'log':c.record(folder/(name+'.log'))}
  c.write(folder/(name+'_execution.json'),item);done.append(item)
  c.need(r.returncode==0,'Teacher step failed: '+name)
  print('Completed '+name,flush=True)
except Exception:error=traceback.format_exc();raise
finally:c.write(folder/'execution.json',{'exit_code':int(error is not None),'failure':error,'steps':done,'elapsed_seconds':time.monotonic()-start,'finished_at_utc':c.now(),'automatic_retry':False})
