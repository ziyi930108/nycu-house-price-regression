"""Freeze, one gated full fit, then artifact CSV release; stop on any failure."""
from pathlib import Path
import subprocess
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'src'))
import round100_contract as c
BASE=c.ROOT/'outputs/research_resume_20261008'
for name,flag in [('final_freeze','--freeze'),('full_fit','--fit'),('csv_release','--release')]:
    tick=time.monotonic()
    cmd=[str(c.ROOT/'.venv/Scripts/python.exe'),'-u','-B','-X','utf8',str(c.ROOT/'src/round107_full_release.py'),flag]
    with (BASE/(name+'.log')).open('x',encoding='utf8') as log:
        p=subprocess.run(cmd,cwd=c.ROOT,stdout=log,stderr=subprocess.STDOUT,check=False)
    c.write(BASE/(name+'_execution.json'),{'exit_code':p.returncode,'elapsed_seconds':time.monotonic()-tick,'command':cmd})
    print(name,p.returncode,round(time.monotonic()-tick,3),flush=True)
    if p.returncode:raise SystemExit(p.returncode)
