import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
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
const tableOwners=[3,4,8,13,16,17,23];
const result=await finalizePresentation({workspaceDir:ROOT,explicitTotalSlideCount:24,candidatePath:QA+'/candidate.pptx',finalPath:final,pythonExecutable:PY,integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[10,11,19],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'design',families:['Microsoft JhengHei','Consolas']},verifyArtifactToolImport:true,receiptPath:QA+`/final_${revision}.validation.json`});
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
