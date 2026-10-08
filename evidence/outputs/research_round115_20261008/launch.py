"""Measured CPU pure-X audit commands; preserve first attempt only."""
from pathlib import Path
import argparse
import json
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = Path(__file__).resolve().parent
if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('freeze', 'run'))
    phase = p.parse_args().phase
    command = [str(ROOT / '.venv/Scripts/python.exe'), '-u', '-B', '-X', 'utf8',
               str(ROOT / 'src/round115_prior_support_x_audit.py'), '--' + phase]
    tick = time.monotonic()
    with (BASE / (phase + '.log')).open('x', encoding='utf-8') as log:
        result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
    with (BASE / (phase + '_execution.json')).open('x', encoding='utf-8') as stream:
        json.dump({'command': command, 'exit_code': result.returncode,
                   'elapsed_seconds': time.monotonic() - tick, 'new_fits': 0, 'GPU_calls': 0},
                  stream, ensure_ascii=False, indent=2)
        stream.write('\n')
    print(phase, result.returncode, flush=True)
    sys.exit(result.returncode)
