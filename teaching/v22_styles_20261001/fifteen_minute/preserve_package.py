"""Use Artifact-authored notes while retaining every unrelated original OPC part."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import json,hashlib,sys
HERE=Path(__file__).resolve().parent;OLD=HERE.parent/'v22_teaching_refinement_20261001'
key=sys.argv[1];entry=next(x for x in json.loads((HERE/'content/notes_plan.json').read_text(encoding='utf-8'))['styles'] if x['style']==key)
source=OLD/'deliverables'/entry['source_file'];build=HERE/'build'/key;draft=build/'candidate.pptx';out=build/'candidate_preserved.pptx'
assert hashlib.sha256(source.read_bytes()).hexdigest()==entry['source_sha256']
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def txt(b):return '\n'.join(t.text or '' for t in E.fromstring(b).findall('.//a:t',ns))
replace={f'ppt/notesSlides/notesSlide{i}.xml' for i in range(1,21)}
with ZipFile(source) as src,ZipFile(draft) as edit,ZipFile(out,'x') as dest:
 for info in src.infolist():
  blob=edit.read(info.filename) if info.filename in replace else src.read(info.filename)
  if info.filename in replace:
   number=int(info.filename.rsplit('notesSlide',1)[1].split('.')[0]);assert txt(blob)==entry['notes'][number-1]
  dest.writestr(info,blob)
with ZipFile(out) as n,ZipFile(source) as o:
 assert n.testzip() is None and n.namelist()==o.namelist()
 changed=[p for p in n.namelist() if n.read(p)!=o.read(p)];assert set(changed)==replace
 receipt={'style':key,'method':'Retain original OPC package; replace only twenty notesSlide XML parts authored by Artifact Tool. No workbook snapshot regeneration.','source_sha256':entry['source_sha256'],'artifact_candidate_sha256':hashlib.sha256(draft.read_bytes()).hexdigest(),'preserved_candidate_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'only_changed_parts':changed,'unchanged_part_count':len(n.namelist())-20,'original_embedded_workbooks_preserved':True}
with (build/'preservation_receipt.json').open('x',encoding='utf-8') as f:json.dump(receipt,f,ensure_ascii=False,indent=2)
print(json.dumps(receipt,ensure_ascii=False))
