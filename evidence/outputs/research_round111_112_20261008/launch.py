"""Formal CPU checks, exact independent source freeze, then six serial fits."""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c

BASE=c.ROOT/'outputs/research_round111_112_20261008'
CPU=c.ROOT/'.venv/Scripts/python.exe'
GPU=c.ROOT/'.venv_torch/Scripts/python.exe'


def run(name,cmd,tested_entry=None):
    before=c.source_hashes(tested_entry) if tested_entry else None
    started=time.monotonic()
    with (BASE/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run([str(x) for x in cmd],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    elapsed=time.monotonic()-started
    result={'exit_code':p.returncode,'elapsed_seconds':elapsed,'command':[str(x) for x in cmd]}
    if tested_entry:
        c.need(before==c.source_hashes(tested_entry),'Selfcheck sources mutated while running')
        result['source_sha256']=before
    c.write(BASE/(name+'_execution.json'),result)
    print(name,p.returncode,round(elapsed,3),flush=True)
    if p.returncode:raise SystemExit(p.returncode)


if __name__=='__main__':
    run('R111_residual_selfcheck',[GPU,'-B','-X','utf8',c.ROOT/'src/round111_residual_prior_train.py','--cpu-unit-check'],
        'src/round111_residual_prior_train.py')
    run('R112_helper_selfcheck',[CPU,'-B','-X','utf8',c.ROOT/'src/round112_ordered_unit.py','--self-check'],
        'src/round112_ordered_unit.py')
    run('freeze',[CPU,'-u','-B','-X','utf8',c.ROOT/'src/round111_112_research.py','--freeze'])
    run('six_fits',[CPU,'-u','-B','-X','utf8',c.ROOT/'src/round111_112_research.py','--train'])
