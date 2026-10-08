"""Measured, non-overwriting commands for the three-fit offset validation."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c

BASE=c.ROOT/'outputs/research_round113_20261008'
CPU=c.ROOT/'.venv/Scripts/python.exe'
GPU=c.ROOT/'.venv_torch/Scripts/python.exe'
DRIVER=c.ROOT/'src/round113_offset_research.py'


def run(name,command,entry=None):
    before=c.source_hashes(entry) if entry else None;tick=time.monotonic()
    with (BASE/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run([str(x) for x in command],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    result={'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick,'command':[str(x) for x in command]}
    if entry:
        c.need(before==c.source_hashes(entry),'Selfcheck sources changed during execution')
        result['source_sha256']=before
    c.write(BASE/(name+'_execution.json'),result)
    print(name,p.returncode,round(result['elapsed_seconds'],3),flush=True)
    if p.returncode:raise SystemExit(p.returncode)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('phase',choices=('checks','freeze','train','evaluate'));a=p.parse_args()
    if a.phase=='checks':
        for name,python,entry,flag in (
            ('R107_loss_selfcheck',GPU,'src/round107_weighted_smooth_train.py','--cpu-unit-check'),
            ('R111_residual_selfcheck',GPU,'src/round111_residual_prior_train.py','--cpu-unit-check'),
            ('R112_helper_selfcheck',CPU,'src/round112_ordered_unit.py','--self-check'),
            ('R113_driver_selfcheck',CPU,'src/round113_offset_research.py','--self-check')):
            run(name,[python,'-B','-X','utf8',c.safe(entry),flag],entry)
    else:
        run(a.phase,[CPU,'-u','-B','-X','utf8',DRIVER,'--'+a.phase])
