"""Coordinator-only measured checks/freeze/fits/evaluation; no overwrite."""
from pathlib import Path
import argparse
import subprocess
import sys
import time
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / 'src'))
import round100_contract as c

BASE = c.ROOT / 'outputs/research_round116_20261008'
CPU = c.ROOT / '.venv/Scripts/python.exe'
DRIVER = 'src/round116_research.py'


def run(name, command, entry=None):
    before = c.source_hashes(entry) if entry else None
    tick = time.monotonic()
    with (BASE / (name + '.log')).open('x', encoding='utf-8') as log:
        process = subprocess.run([str(x) for x in command], cwd=c.ROOT, stdout=log, stderr=subprocess.STDOUT, check=False)
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
    p = argparse.ArgumentParser()
    p.add_argument('phase', choices=('checks', 'freeze', 'train', 'evaluate'))
    phase = p.parse_args().phase
    if phase == 'checks':
        for name, entry in [('R116_helper_selfcheck', 'src/round116_depthwise_unit.py'),
                            ('R116_driver_selfcheck', DRIVER)]:
            run(name, [CPU, '-B', '-X', 'utf8', c.safe(entry), '--self-check'], entry)
    else:
        run(phase, [CPU, '-u', '-B', '-X', 'utf8', c.safe(DRIVER), '--' + phase])
