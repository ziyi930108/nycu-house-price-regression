"""Rebuild frozen spoken-note decks from repository inputs in a new directory.

Uses the provided Codex presentation runtime. No model/data inputs or downloads.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import hashlib,json,os,posixpath,re,shutil,subprocess,time
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E

TEACHING=Path('teaching/v22_styles_20261001')
NAMES={'A':'白底學術','B':'深色科技','C':'暖色雜誌'}
NS={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main','r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
EDITED={f'ppt/notesSlides/notesSlide{i}.xml' for i in range(1,21)}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def texts(b):return '\n'.join(t.text or '' for t in E.fromstring(b).findall('.//a:t',NS))
def once(s,old,new):
 if s.count(old)!=1:raise ValueError('Frozen source anchor changed: '+old)
 return s.replace(old,new,1)
def assignment(s,pattern,new):
 value,n=re.subn(pattern,lambda _:new,s)
 if n!=1:raise ValueError('Frozen source assignment changed: '+pattern)
 return value
def related(z,part):
 parent=posixpath.dirname(part);rp=parent+'/_rels/'+posixpath.basename(part)+'.rels'
 return {r.attrib['Id']:(r.attrib['Type'],posixpath.normpath(posixpath.join(parent,r.attrib['Target'])).lstrip('/')) for r in E.fromstring(z.read(rp)) if r.attrib.get('TargetMode')!='External'}
def note_order(z):
 rels=related(z,'ppt/presentation.xml');order=[]
 for item in E.fromstring(z.read('ppt/presentation.xml')).findall('p:sldIdLst/p:sldId',NS):
  kind,slide=rels[item.attrib['{'+NS['r']+'}id']]
  if not kind.endswith('/slide'):raise ValueError('Unexpected slide relationship')
  notes=[target for kind,target in related(z,slide).values() if kind.endswith('/notesSlide')]
  if len(notes)!=1:raise ValueError('Expected one notes relationship per slide')
  order.append(notes[0])
 if order!=[f'ppt/notesSlides/notesSlide{i}.xml' for i in range(1,24)]:raise ValueError('Frozen slide/notes order changed')
 return order
def preserve(source,draft,target,notes):
 with ZipFile(source) as o,ZipFile(draft) as n,ZipFile(target,'x') as out:
  note_order(o);note_order(n)
  for item in o.infolist():
   blob=n.read(item.filename) if item.filename in EDITED else o.read(item.filename)
   if item.filename in EDITED:
    i=int(item.filename.rsplit('notesSlide',1)[1].split('.')[0]);assert texts(blob)==notes[i-1]
   out.writestr(item,blob)
def verify_package(source,target,notes):
 with ZipFile(source) as o,ZipFile(target) as n:
  assert n.testzip() is None
  assert len(n.namelist())==len(set(n.namelist())) and set(n.namelist())==set(o.namelist())
  order=note_order(n)
  assert {p for p in n.namelist() if n.read(p)!=o.read(p)}==EDITED
  assert all(texts(n.read(part))==notes[i] for i,part in enumerate(order))
 return order

def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--repository',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output',type=Path,required=True)
 for n in ('node','python','node-modules','presentation-skill'):ap.add_argument('--'+n,type=Path,required=True)
 ap.add_argument('--styles',nargs='+',choices=NAMES,default=list(NAMES));ap.add_argument('--check-only',action='store_true');args=ap.parse_args()
 if len(args.styles)!=len(set(args.styles)):ap.error('Duplicate styles')
 if os.path.lexists(args.output):ap.error('Output must not exist')
 repo,out=args.repository.resolve(),args.output.resolve()
 node,python,modules,skill=(p.resolve() for p in (args.node,args.python,args.node_modules,args.presentation_skill))
 for p in (repo,modules,skill,node.parent,python.parent):
  if out.is_relative_to(p) or p.is_relative_to(out):raise ValueError('Output overlaps protected source/runtime')
 for p in [node,python,modules/'@oai/artifact-tool/package.json']+[skill/'container_tools'/n for n in ('artifact_tool_utils.mjs','inspect_presentation_package_integrity.py','inspect_presentation_layout_geometry.py')]:
  if not p.is_file():raise ValueError('Missing runtime dependency: '+str(p))
 source=repo/TEACHING;spoken=source/'spoken'
 manifest=repo/'SOURCE_MANIFEST.json';expected={x['path']:x['sha256'] for x in json.loads(manifest.read_text(encoding='utf-8'))['files']}
 inputs=[spoken/'build/edit_notes.mjs',spoken/'build/finalize_notes.mjs',spoken/'delivery_receipt.json']
 content_names=['notes_plan.json','narrative.json','完整口語講稿.md','上台口述提示.md','排練與問答.md']
 inputs += [spoken/'content'/n for n in content_names]
 inputs += [source/'deliverables'/f'臺灣房價預測_v22_{NAMES[k]}_r2.pptx' for k in args.styles]
 hashes={}
 for p in inputs:
  if not p.resolve().is_relative_to(repo):raise ValueError('Source escaped repository')
  relative=p.relative_to(repo).as_posix();value=sha(p)
  if value!=expected.get(relative):raise ValueError('Source identity mismatch: '+relative)
  hashes[relative]=value
 plan=json.loads((spoken/'content/notes_plan.json').read_text(encoding='utf-8'))
 if [x['style'] for x in plan['styles']]!=list(NAMES):raise ValueError('Unexpected notes plan styles')
 entries={x['style']:x for x in plan['styles']}
 for k in args.styles:
  e=entries[k]
  if e['source_file']!=f'臺灣房價預測_v22_{NAMES[k]}_r2.pptx' or len(e['notes'])!=23 or not all(isinstance(x,str) for x in e['notes']):raise ValueError('Invalid frozen notes plan')
  if sha(source/'deliverables'/e['source_file'])!=e['source_sha256']:raise ValueError('Original deck changed')
  with ZipFile(source/'deliverables'/e['source_file']) as z:note_order(z)
 reference=json.loads((spoken/'delivery_receipt.json').read_text(encoding='utf-8'))
 edit=once((spoken/'build/edit_notes.mjs').read_text(encoding='utf-8'),"const OLD=path.resolve(DIR,'../v22_teaching_refinement_20261001');",'const OLD=process.env.TEACHING_SOURCE;')
 final=(spoken/'build/finalize_notes.mjs').read_text(encoding='utf-8')
 final=assignment(final,r"const SKILL='[^'\r\n]+';",'const SKILL=process.env.TEACHING_PRESENTATION_SKILL;')
 final=assignment(final,r"const PY='[^'\r\n]+';",'const PY=process.env.RUNTIME_PYTHON;')
 final=assignment(final,r"process\.env\.RUNTIME_NODE_MODULES='[^'\r\n]+';","if(!process.env.RUNTIME_NODE_MODULES)throw Error('Missing Node modules');")
 final=once(final,"const source=path.resolve(DIR,'../v22_teaching_refinement_20261001/deliverables',entry.source_file);","const source=path.join(process.env.TEACHING_SOURCE,'deliverables',entry.source_file);")
 env=os.environ.copy();env.update({'TEACHING_SOURCE':str(source),'TEACHING_PRESENTATION_SKILL':str(skill),'RUNTIME_PYTHON':str(python),'RUNTIME_NODE_MODULES':str(modules)})
 pre={'repository':str(repo),'output':str(out),'styles':args.styles,'input_sha256':hashes,'source_manifest_sha256':sha(manifest),'runner_sha256':sha(Path(__file__)),'runtime_paths':{k:env[k] for k in ('TEACHING_PRESENTATION_SKILL','RUNTIME_PYTHON','RUNTIME_NODE_MODULES')},'runtime_versions':{'node':subprocess.check_output([str(node),'--version'],text=True).strip(),'python':subprocess.check_output([str(python),'--version'],text=True).strip()},'required_fonts':['Microsoft JhengHei','Consolas']+(['PMingLiU'] if 'C' in args.styles else [])}
 if args.check_only:print(json.dumps({'status':'preflight_passed',**pre},ensure_ascii=False));return
 out.mkdir(parents=True,exist_ok=False);save(out/'preflight.json',pre);started=time.perf_counter();steps=[]
 try:
  for folder in ('build','deliverables','content'):(out/folder).mkdir()
  for name in content_names:shutil.copy2(spoken/'content'/name,out/'content'/name)
  for name,value in [('edit_notes.mjs',edit),('finalize_notes.mjs',final)]:
   with (out/'build'/name).open('x',encoding='utf-8',newline='\n') as f:f.write(value)
  link=out/'build/node_modules'
  if os.name=='nt':
   def quote(v):return "'"+str(v).replace("'","''")+"'"
   subprocess.run(['powershell.exe','-NoProfile','-NonInteractive','-Command','New-Item -ItemType Junction -Path '+quote(link)+' -Target '+quote(modules)+' -ErrorAction Stop | Out-Null'],check=True)
  else:link.symlink_to(modules,target_is_directory=True)
  outputs=[]
  for k in args.styles:
   e=entries[k];build=out/'build'/k;build.mkdir();original=source/'deliverables'/e['source_file']
   for stage in ('edit','preserve','finalize'):
    start=time.perf_counter()
    if stage=='preserve':
     preserve(original,build/'candidate.pptx',build/'candidate_preserved.pptx',e['notes']);code=0
    else:
     command=[str(node),str(out/'build'/('edit_notes.mjs' if stage=='edit' else 'finalize_notes.mjs')),k]+(['preserved'] if stage=='finalize' else [])
     with (build/(stage+'.stdout.log')).open('xb') as stdout,(build/(stage+'.stderr.log')).open('xb') as stderr:
      code=subprocess.run(command,cwd=out,env=env,stdout=stdout,stderr=stderr).returncode
    steps.append({'style':k,'stage':stage,'exit_code':code,'elapsed_seconds':time.perf_counter()-start});print(json.dumps(steps[-1]),flush=True)
    if code:raise RuntimeError('Build failed; inspect saved log and use a new output directory')
   deck=out/'deliverables'/e['source_file'].replace('_r2.pptx','_口語講稿版.pptx')
   order=verify_package(original,deck,e['notes'])
   validation=json.loads((build/'final_preserved.validation.json').read_text(encoding='utf-8'))
   assert validation['finalSha256']==sha(deck) and validation['packageIntegrity']['status']=='pass'
   renders={f'slide-{i:02}':sha(build/'renders'/f'slide-{i:02}.png') for i in range(1,24)}
   prior=next(x for x in reference['styles'] if x['style']==k)
   if renders!=prior['render_sha256']:raise ValueError('Rendered pixels differ from frozen edition; inspect runtime and output, do not claim visual identity')
   outputs.append({'style':k,'path':deck.relative_to(out).as_posix(),'sha256':sha(deck),'original_spoken_sha256':prior['sha256'],'pptx_bytes_identical':sha(deck)==prior['sha256'],'all_23_render_bytes_identical':True,'render_sha256':renders,'all_23_notes_match':True,'only_main_notes_parts_changed':True,'notes_in_actual_slide_order':order})
  assert all(sha(repo/p)==h for p,h in hashes.items())
  save(out/'rebuild_receipt.json',{'status':'completed','completed_at_utc':datetime.now(timezone.utc).isoformat(),'elapsed_seconds':time.perf_counter()-started,'steps':steps,'outputs':outputs,'all_project_inputs_from_repository':True,'frozen_sources_unchanged':True,'new_ML_fits':0,'clean_runtime_installation_tested':False,'PowerPoint_desktop_tested':False})
 except Exception as exc:
  save(out/'failure.json',{'status':'failed','error':str(exc),'steps':steps,'elapsed_seconds':time.perf_counter()-started});raise

if __name__=='__main__':main()
