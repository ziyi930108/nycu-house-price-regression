"""Coordinator-only measured checks, freeze, four serial fits, separate scoring."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import round100_contract as c

BASE = c.ROOT / 'outputs/research_round117_v2_20261008'
DRIVER = 'src/round117_research_v2.py'


def run(name, command, entry=None):
    before = c.source_hashes(entry) if entry else None
    tick = time.monotonic()
    with (BASE / (name + '.log')).open('x', encoding='utf-8') as log:
        process = subprocess.run([str(x) for x in command], cwd=c.ROOT,
                                 stdout=log, stderr=subprocess.STDOUT, check=False)
    receipt = {'exit_code': process.returncode, 'elapsed_seconds': time.monotonic() - tick,
               'command': [str(x) for x in command]}
    if entry:
        c.need(before == c.source_hashes(entry), 'Formal selfcheck source changed')
        receipt['source_sha256'] = before
    c.write(BASE / (name + '_execution.json'), receipt)
    print(name, process.returncode, round(receipt['elapsed_seconds'], 3), flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('checks', 'freeze', 'train', 'evaluate'))
    phase = parser.parse_args().phase
    if phase == 'checks':
        for name, entry, environment, flag in (
            ('R117_prior_selfcheck', 'src/round117_transaction_prior.py', '.venv', '--self-check'),
            ('R117_trainer_selfcheck', 'src/round117_transaction_prior_train.py', '.venv_torch', '--cpu-unit-check'),
            ('R117_driver_selfcheck', DRIVER, '.venv', '--self-check')):
            run(name, [c.ROOT / environment / 'Scripts/python.exe', '-B', '-X', 'utf8', c.safe(entry), flag], entry)
    else:
        run(phase, [c.ROOT / '.venv/Scripts/python.exe', '-u', '-B', '-X', 'utf8', c.safe(DRIVER), '--' + phase])
