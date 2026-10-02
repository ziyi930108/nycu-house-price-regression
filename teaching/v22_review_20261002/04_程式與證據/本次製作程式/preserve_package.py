"""Retain original OPC parts and take only selected Artifact-authored notes."""
from __future__ import annotations
import argparse, hashlib, json, posixpath
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E

NS = {'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
      'p': 'http://schemas.openxmlformats.org/presentationml/2006/main',
      'r': 'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def save(p, value):
    with Path(p).open('x', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write('\n')

def text(blob):
    return '\n'.join(t.text or '' for t in E.fromstring(blob).findall('.//a:t', NS))

def related(z, part):
    folder = posixpath.dirname(part)
    rel = folder + '/_rels/' + posixpath.basename(part) + '.rels'
    result = {}
    for item in E.fromstring(z.read(rel)):
        if item.attrib.get('TargetMode') == 'External':
            continue
        target = item.attrib['Target']
        target = target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(folder, target))
        if target.startswith('../'):
            raise ValueError('Relationship escapes the OPC package')
        result[item.attrib['Id']] = (item.attrib['Type'], target)
    return result

def note_order(z):
    rels = related(z, 'ppt/presentation.xml')
    result = []
    for item in E.fromstring(z.read('ppt/presentation.xml')).findall('p:sldIdLst/p:sldId', NS):
        kind, slide = rels[item.attrib['{' + NS['r'] + '}id']]
        if not kind.endswith('/slide'):
            raise ValueError('Unexpected slide relationship')
        notes = [p for k, p in related(z, slide).values() if k.endswith('/notesSlide')]
        if len(notes) != 1:
            raise ValueError('Expected one notes relationship per slide')
        result.append(notes[0])
    if len(result) != 23 or len(set(result)) != 23:
        raise ValueError('Expected 23 distinct notes parts')
    return result

def verify(source, target, entry):
    with ZipFile(source) as old, ZipFile(target) as new:
        if new.testzip() is not None or len(new.namelist()) != len(set(new.namelist())):
            raise ValueError('Invalid or duplicated final ZIP members')
        if old.namelist() != new.namelist():
            raise ValueError('Final OPC member order or identity changed')
        order = note_order(old)
        if order != note_order(new):
            raise ValueError('Final slide-to-notes relationships changed')
        expected = {order[i - 1] for i in entry['changed_slide_numbers']}
        changed = [p for p in new.namelist() if new.read(p) != old.read(p)]
        if set(changed) != expected:
            raise ValueError('Changes extend beyond declared notes parts')
        for i, part in enumerate(order):
            if text(new.read(part)) != entry['notes'][i]:
                raise ValueError('Final note text mismatch on slide ' + str(i + 1))
        workbook_paths = [p for p in old.namelist() if p.startswith('ppt/embeddings/')]
        return {'only_changed_parts': changed, 'notes_in_slide_order': order,
                'all_23_notes_match': True, 'unchanged_part_count': len(new.namelist()) - len(changed),
                'original_embedded_workbooks_preserved': True,
                'embedded_workbook_sha256': {p: hashlib.sha256(old.read(p)).hexdigest() for p in workbook_paths}}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-dir', type=Path, required=True)
    parser.add_argument('--work-dir', type=Path, required=True)
    parser.add_argument('--style', choices=['A', 'B', 'C'], required=True)
    args = parser.parse_args()
    entry = next(x for x in json.loads((args.work_dir / 'content/notes_plan.json').read_text(encoding='utf-8'))['styles'] if x['style'] == args.style)
    source = args.source_dir / entry['source_file']
    if sha(source) != entry['source_sha256']:
        raise ValueError('Source identity changed')
    build = args.work_dir / 'build' / args.style
    draft, target = build / 'candidate.pptx', build / 'candidate_preserved.pptx'
    with ZipFile(source) as old, ZipFile(draft) as edited, ZipFile(target, 'x') as out:
        order = note_order(old)
        if order != note_order(edited):
            raise ValueError('Artifact import changed slide-to-notes relationship identities')
        replacements = {order[i - 1] for i in entry['changed_slide_numbers']}
        for info in old.infolist():
            out.writestr(info, edited.read(info.filename) if info.filename in replacements else old.read(info.filename))
    receipt = {'style': args.style, 'source_sha256': sha(source),
               'artifact_candidate_sha256': sha(draft), 'preserved_candidate_sha256': sha(target),
               'method': 'Original OPC package with only declared notes XML from Artifact Tool; no workbook regeneration.',
               **verify(source, target, entry)}
    save(build / 'preservation_receipt.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False))

if __name__ == '__main__':
    main()
