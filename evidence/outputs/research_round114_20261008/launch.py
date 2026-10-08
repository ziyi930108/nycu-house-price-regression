"""Measured source checks and coordinator-only R114 commands, no overwrites."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c
BASE=c.ROOT/'outputs/research_round114_20261008'
CPU=c.ROOT/'.venv/Scripts/python.exe'
GPU=c.ROOT/'.venv_torch/Scripts/python.exe'
DRIVER=c.ROOT/'src/round114_research.py'


def run(name,command,entry=None):
    before=c.source_hashes(entry) if entry else None;tick=time.monotonic()
    with (BASE/(name+'.log')).open('x',encoding='utf8') as log:
        process=subprocess.run([str(x) for x in command],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    result={'exit_code':process.returncode,'elapsed_seconds':time.monotonic()-tick,'command':[str(x) for x in command]}
    if entry:
        c.need(before==c.source_hashes(entry),'Selfcheck source closure changed')
        result['source_sha256']=before
    c.write(BASE/(name+'_execution.json'),result)
    print(name,process.returncode,round(result['elapsed_seconds'],3),flush=True)
    if process.returncode:raise SystemExit(process.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('checks','freeze','train','evaluate'));a=p.parse_args()
    if a.phase=='checks':
        for name,python,entry,flag in (
            ('R114_trainer_selfcheck',GPU,'src/round114_additive_prior_train.py','--cpu-unit-check'),
            ('R114_driver_selfcheck',CPU,'src/round114_research.py','--self-check')):
            run(name,[python,'-B','-X','utf8',c.safe(entry),flag],entry)
    else:run(a.phase,[CPU,'-u','-B','-X','utf8',DRIVER,'--'+a.phase])
