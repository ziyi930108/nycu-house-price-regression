"""One CPU audit attempt after root confirmed separate formal evaluate exit0."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import round100_contract as c
FOLDER='outputs/research_round114_20261008/'
TOOL=FOLDER+'verify_completed.py';LOG=FOLDER+'postfit_review.log'
RECEIPT=FOLDER+'independent_postfit_execution.json'
c.need(c.sha(TOOL)=='72929eff3901adb1592a5dbf2415c9cc0118c6293eb23c0a28346a48aae3868d','Prepared tool changed')
for name in (LOG,RECEIPT,str(Path(RECEIPT).with_suffix('.sha256'))):
    c.need(not c.safe(name).exists(),'Preserve prior independent attempt')
command=[str(ROOT/'.venv/Scripts/python.exe'),'-B','-X','utf8',str(c.safe(TOOL)),'--evaluation-complete']
tick=time.monotonic();code=None;failure=None
try:
    with c.safe(LOG).open('x',encoding='utf8',newline='\n') as output:
        result=subprocess.run(command,cwd=ROOT,stdout=output,stderr=subprocess.STDOUT,
            env={**os.environ,'CUDA_VISIBLE_DEVICES':'','PYTHONDONTWRITEBYTECODE':'1'})
        code=result.returncode
except BaseException as exc:
    failure=type(exc).__name__+': '+str(exc)
payload={'status':'independent_CPU_postfit_execution','command':command,'exit_code':code,'failure':failure,
         'elapsed_seconds':time.monotonic()-tick,'review_tool':c.record(TOOL),'launcher':c.record(__file__),
         'log':c.record(LOG),'evaluation_execution':c.record(FOLDER+'evaluation_execution.json'),
         'new_model_fits':0,'GPU_work':False,'automatic_retry':False}
if c.safe(FOLDER+'postfit_review.json').exists():payload['postfit_review']=c.record(FOLDER+'postfit_review.json')
c.write(RECEIPT,payload)
print(json.dumps(payload,ensure_ascii=False))
sys.exit(code if code is not None else 1)
