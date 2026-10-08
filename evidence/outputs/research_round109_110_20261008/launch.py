"""Formal CPU checks, source/input freeze, then six serial true fits."""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c

BASE=c.ROOT/'outputs/research_round109_110_20261008'
CPU=c.ROOT/'.venv/Scripts/python.exe'
GPU=c.ROOT/'.venv_torch/Scripts/python.exe'

def run(name,cmd):
    started=time.monotonic()
    with (BASE/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run([str(x) for x in cmd],cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    elapsed=time.monotonic()-started
    c.write(BASE/(name+'_execution.json'),{'exit_code':p.returncode,'elapsed_seconds':elapsed,'command':[str(x) for x in cmd]})
    print(name,p.returncode,round(elapsed,3),flush=True)
    if p.returncode:raise SystemExit(p.returncode)

run('R109_average_selfcheck',[GPU,'-B','-X','utf8',c.ROOT/'src/round109_tail_parameter_average_train.py','--cpu-unit-check'])
run('R110_helper_selfcheck',[CPU,'-B','-X','utf8',c.ROOT/'src/round110_target_borders_unit.py','--self-check'])
run('freeze',[CPU,'-u','-B','-X','utf8',c.ROOT/'src/round109_110_research.py','--freeze'])
run('six_fits',[CPU,'-u','-B','-X','utf8',c.ROOT/'src/round109_110_research.py','--train'])
