"""One exclusive CPU audit invocation, after root confirmed evaluation exit 0."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
GROUP = ROOT / 'outputs/research_round111_112_20261008'
TOOL = GROUP / 'verify_completed.py'
LOG = GROUP / 'independent_postfit.log'
RECEIPT = GROUP / 'independent_postfit_execution.json'
EXPECTED = 'db1ca660541265b16482bb17ba8f0696c5e2201fa9fec3c7c7ccdbe353b8afc8'

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def record(path):
    return {'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)}

def main():
    assert sha(TOOL) == EXPECTED, 'Prepared independent audit tool changed'
    for path in (LOG, RECEIPT, RECEIPT.with_suffix('.sha256')):
        assert not path.exists(), 'Preserve previous audit attempt: ' + str(path)
    command = [str(ROOT / '.venv/Scripts/python.exe'), '-B', '-X', 'utf8', str(TOOL), '--evaluation-complete']
    start = time.monotonic()
    code = None
    failure = None
    try:
        with LOG.open('x', encoding='utf-8', newline='\n') as output:
            completed = subprocess.run(command, cwd=ROOT, stdout=output, stderr=subprocess.STDOUT,
                                       env={**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'CUDA_VISIBLE_DEVICES': ''})
            code = completed.returncode
    except BaseException as exc:
        failure = type(exc).__name__ + ': ' + str(exc)
    payload = {'status': 'independent_CPU_postfit_execution', 'command': command,
               'exit_code': code, 'failure': failure, 'elapsed_seconds': time.monotonic() - start,
               'review_tool': record(TOOL), 'launcher': record(Path(__file__)), 'log': record(LOG),
               'evaluation_execution': record(GROUP / 'evaluation_execution.json'),
               'new_model_fits': 0, 'GPU_work': False, 'automatic_retry': False}
    review = GROUP / 'postfit_review.json'
    if review.exists():
        payload['postfit_review'] = record(review)
    with RECEIPT.open('x', encoding='utf-8', newline='\n') as output:
        json.dump(payload, output, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        output.write('\n')
    with RECEIPT.with_suffix('.sha256').open('x', encoding='ascii', newline='\n') as output:
        output.write(sha(RECEIPT) + '  ' + RECEIPT.name + '\n')
    print(json.dumps(payload, ensure_ascii=False))
    return code if code is not None else 1

if __name__ == '__main__':
    sys.exit(main())
