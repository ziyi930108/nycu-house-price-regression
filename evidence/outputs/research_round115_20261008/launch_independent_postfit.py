"""Preserve a single bounded CPU aggregate audit attempt; no retry."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import round100_contract as c
FOLDER = 'outputs/research_round115_20261008/'
TOOL = FOLDER + 'verify_completed.py'
LOG = FOLDER + 'independent_postfit.log'
RECEIPT = FOLDER + 'independent_postfit_execution.json'
for name in (LOG, RECEIPT, str(Path(RECEIPT).with_suffix('.sha256'))):
    c.need(not c.safe(name).exists(), 'Preserve prior bounded audit attempt')
command = [str(ROOT / '.venv/Scripts/python.exe'), '-B', '-X', 'utf8', str(c.safe(TOOL))]
source = c.record(TOOL)
compile(c.safe(TOOL).read_text(encoding='utf8'), TOOL, 'exec')
tick = time.monotonic()
with c.safe(LOG).open('x', encoding='utf8', newline='\n') as log:
    result = subprocess.run(command, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
        env={**os.environ, 'CUDA_VISIBLE_DEVICES': '', 'PYTHONDONTWRITEBYTECODE': '1'})
payload = {'status': 'independent_bounded_CPU_aggregate_review_execution', 'command': command,
    'exit_code': result.returncode, 'elapsed_seconds': time.monotonic() - tick,
    'review_tool_before': source, 'review_tool_after': c.record(TOOL),
    'launcher': c.record(__file__), 'log': c.record(LOG), 'new_fits': 0,
    'GPU_work': False, 'automatic_retry': False, 'query_y_or_housing_arrays_deserialized': False}
if c.safe(FOLDER + 'postfit_review.json').exists(): payload['postfit_review'] = c.record(FOLDER + 'postfit_review.json')
c.write(RECEIPT, payload)
print(json.dumps(payload, ensure_ascii=False))
sys.exit(result.returncode)
