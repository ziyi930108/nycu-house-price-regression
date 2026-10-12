import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {spawnSync} from 'node:child_process';
const MODULES=process.env.RUNTIME_NODE_MODULES||'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const {FileBlob,PresentationFile}=await import(pathToFileURL(path.join(MODULES,'@oai/artifact-tool/dist/artifact_tool.mjs')).href);
const QA=path.resolve(process.argv[2]||path.join(import.meta.dirname,'..','rebuild'));
const ROOT=QA;
const SKILL=process.env.PRESENTATIONS_SKILL_DIR||'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const PY=process.env.RUNTIME_PYTHON||'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES=MODULES;
const revision=process.argv[3]||'r1';
if(!/^r\d+$/.test(revision))throw Error('Revision required');
const final=QA+`/final/${revision}.pptx`, renders=QA+`/renders_${revision}`;
await fs.mkdir(QA+'/final',{recursive:true});
const {finalizePresentation}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
const tableOwners=[4,5,6,7,8,12];
// Normalize optional package counters only; authored slide content stays intact.
const metadataPath=QA+`/candidate_metadata_${revision}.pptx`;
const metadataCode=`import sys,zipfile,copy
from lxml import etree as E
src,dest=sys.argv[1:]
ns='http://schemas.openxmlformats.org/officeDocument/2006/extended-properties'
with zipfile.ZipFile(src) as z,zipfile.ZipFile(dest,'x',compression=zipfile.ZIP_DEFLATED) as out:
 for info in z.infolist():
  data=z.read(info.filename)
  if info.filename=='docProps/app.xml':
   root=E.fromstring(data)
   for tag in ['Slides','Notes']:
    node=root.find('{'+ns+'}'+tag)
    if node is None: node=E.SubElement(root,'{'+ns+'}'+tag)
    node.text='16'
   data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
  out.writestr(copy.copy(info),data)
`;
const metadata=spawnSync(PY,['-c',metadataCode,QA+'/candidate.pptx',metadataPath],{encoding:'utf8'});
if(metadata.status!==0)throw Error(metadata.stderr||'Metadata normalization failed');
const result=await finalizePresentation({workspaceDir:ROOT,explicitTotalSlideCount:16,candidatePath:metadataPath,finalPath:final,pythonExecutable:PY,integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[9,10,15],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Microsoft JhengHei'],referencePath:'C:/NYCU_ML/deliverables/房屋價格預測_日期與參考價教學_20261012/台灣房屋銷售價格預測挑戰賽_上台簡報.pptx',referenceSha256:'db3331a785219760a584c32cdd5533871ea65a1ab2ff290c54c9ba3d803f5afb'},verifyArtifactToolImport:true,receiptPath:QA+`/final_${revision}.validation.json`});
console.log(JSON.stringify({event:'finalized',file:final,status:result.status}));
await fs.mkdir(renders,{recursive:true});
const p=await PresentationFile.importPptx(await FileBlob.load(final));
for(let i=0;i<p.slides.items.length;i++){
  const slide=p.slides.items[i],stem='slide-'+String(i+1).padStart(2,'0');
  const png=await p.export({slide,format:'png',scale:1.25});
  await fs.writeFile(path.join(renders,stem+'.png'),new Uint8Array(await png.arrayBuffer()),{flag:'wx'});
  const layout=await slide.export({format:'layout'});
  await fs.writeFile(path.join(renders,stem+'.json'),await layout.text(),{flag:'wx'});
  console.log(JSON.stringify({event:'rendered',slide:i+1}));
}
console.log(JSON.stringify({status:'requires_visual_review',final,renders}));
