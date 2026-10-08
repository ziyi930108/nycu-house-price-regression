"""One CPU R117 engineering-v2 audit, only after root confirms evaluate exit0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
FOLDER='outputs/research_round117_v2_20261008/'
TOOL=FOLDER+'verify_completed_v2_roots.py'
NATIVE=FOLDER+'audit_model_identity_v2.py'
LOG=FOLDER+'postfit_review.log'
RECEIPT=FOLDER+'independent_postfit_execution.json'
c.need(c.sha(TOOL)=='cd6504c3c4c06d87fa84d9cd5a8793ede2a312bb99febe8396665172a2e32a39','Prepared audit tool changed')
c.need(c.sha(NATIVE)=='3b69f3f49498d0fec0fe988d313ec8572c0edfa0937a8af834e683590f16581f','Prepared native audit tool changed')
for name in (LOG,RECEIPT,str(Path(RECEIPT).with_suffix('.sha256')),FOLDER+'postfit_review.json'):
    c.need(not c.safe(name).exists(),'Preserve prior independent attempt')
c.need(c.read(FOLDER+'evaluation_execution.json')['exit_code']==0
       and c.read(FOLDER+'training_execution.json')['exit_code']==0
       and c.safe('outputs/round117_v2/seal.json').exists(),'Formal separate four-fit seal and evaluation exit0 required')
command=[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(c.safe(TOOL)),'--evaluation-complete']
tick=time.monotonic();code=None;failure=None
try:
    with c.safe(LOG).open('x',encoding='utf8',newline='\n') as output:
        process=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        code=process.returncode
except BaseException as exc:
    failure=type(exc).__name__+': '+str(exc)
payload={'status':'independent_CPU_postfit_execution','command':command,'exit_code':code,'failure':failure,
    'elapsed_seconds':time.monotonic()-tick,'review_tool':c.record(TOOL),'launcher':c.record(__file__),
    'log':c.record(LOG),'evaluation_execution':c.record(FOLDER+'evaluation_execution.json'),
    'new_model_fits':0,'GPU_work':False,'automatic_retry':False,
    'explicit_engineering_recovery_v2':True}
if c.safe(FOLDER+'postfit_review.json').exists():payload['postfit_review']=c.record(FOLDER+'postfit_review.json')
c.write(RECEIPT,payload)
print(json.dumps(payload,ensure_ascii=False))
sys.exit(code if code is not None else 1)
