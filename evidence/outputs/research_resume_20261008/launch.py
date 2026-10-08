"""Record preflight stdout and launch one serial GPU research driver."""
from pathlib import Path
import hashlib
import json
import subprocess
import time

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).parent

def run(command,name):
    start=time.monotonic()
    with (OUT/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    receipt={'command':command,'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-start,
             'log':str((OUT/(name+'.log')).relative_to(ROOT)),
             'log_sha256':hashlib.sha256((OUT/(name+'.log')).read_bytes()).hexdigest()}
    with (OUT/(name+'_execution.json')).open('x',encoding='utf8') as f:json.dump(receipt,f,indent=2)
    if p.returncode:raise RuntimeError('Failed; preserve log and stop: '+name)
    print(json.dumps({'completed':name,'elapsed_seconds':receipt['elapsed_seconds']}),flush=True)

cpu=str(ROOT/'.venv/Scripts/python.exe');gpu=str(ROOT/'.venv_torch/Scripts/python.exe')
review=json.loads((OUT/'preflight_review.json').read_text(encoding='utf8'))
if review['status']!='passed':raise RuntimeError('Independent preflight not passed')
for name,h in review['source_sha256'].items():
    if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=h:raise RuntimeError('Reviewed source changed: '+name)
run([gpu,'-B','-X','utf8','src/round107_weighted_smooth_train.py','--cpu-unit-check'],'R107_loss_selfcheck')
run([cpu,'-B','-X','utf8','src/round108_single_ctr_unit.py','--self-check'],'R108_helper_selfcheck')
run([cpu,'-B','-X','utf8','src/round107_108_research.py','--freeze'],'freeze')
run([cpu,'-u','-B','-X','utf8','src/round107_108_research.py','--train'],'six_fits')
