"""Build a self-contained rehearsal page without changing the approved script."""
from pathlib import Path
import json, hashlib
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
source=ROOT/'outputs/v22_spoken_20261001/content/narrative.json'
approved=ROOT/'outputs/v22_spoken_20261001/content/approved_scripts.json'
slides=[s for s in json.loads(source.read_text(encoding='utf-8'))['slides'] if not s['is_appendix']]
scripts=json.loads(approved.read_text(encoding='utf-8'))['slides']
assert len(slides)==20
assert [(s['number'],s['script']) for s in slides]==[(s['number'],s['script']) for s in scripts]
payload={'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'slides':[{k:s[k] for k in ('number','title','takeaway','script','seconds')} for s in slides]}
template=(HERE/'template.html').read_text(encoding='utf-8')
assert template.count('__DECK_JSON__')==1
output=HERE/'v22-rehearsal-reviewed.html'
with output.open('x',encoding='utf-8') as f:f.write(template.replace('__DECK_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
receipt={'input':source.relative_to(ROOT).as_posix(),'source_sha256':payload['source_sha256'],'approved_sha256':hashlib.sha256(approved.read_bytes()).hexdigest(),'slides':20,'script_exact_match':True,'output_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),'bytes':output.stat().st_size,'model_training':False,'human_rehearsal_completed':False}
(HERE/'build_reviewed_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(receipt))
