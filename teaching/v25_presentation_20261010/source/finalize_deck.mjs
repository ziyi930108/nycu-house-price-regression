import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const DIR=path.resolve(import.meta.dirname,'..');
const BUILD=path.join(DIR,'build','A'), OUT=path.join(DIR,'deliverables');
const SKILL='C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations';
const PY='C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
process.env.RUNTIME_NODE_MODULES='C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const reference=path.resolve(DIR,'../../deliverables/v22_github/teaching/v22_styles_20261001/deliverables/臺灣房價預測_v22_白底學術_r2.pptx');
const revision=process.argv[2]||'r1';
if(!/^r[1-9][0-9]*$/.test(revision))throw Error('Use a new revision name');
const final=path.join(OUT,`臺灣房價預測_v25_含實際訓練曲線_${revision}.pptx`);
const renders=path.join(BUILD,`renders_${revision}`);
const receipt=path.join(BUILD,`final_${revision}.validation.json`);
for(const f of [final,renders,receipt]){
  try{await fs.access(f);throw Error('Preserve existing output: '+f);}
  catch(e){if(e.code!=='ENOENT')throw e;}
}
const {finalizePresentation}=await import(pathToFileURL(SKILL+'/container_tools/artifact_tool_utils.mjs').href);
await fs.mkdir(renders,{recursive:true}); await fs.mkdir(OUT,{recursive:true});
const tableOwners=[3,4,6,7,8,10,12,13,23,24];
const result=await finalizePresentation({workspaceDir:DIR,explicitTotalSlideCount:24,candidatePath:BUILD+'/candidate.pptx',finalPath:final,pythonExecutable:PY,integrityValidatorPath:SKILL+'/container_tools/inspect_presentation_package_integrity.py',layoutValidatorPath:SKILL+'/container_tools/inspect_presentation_layout_geometry.py',layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n=>['--require-native-table-slide',String(n)])],requiredNativeTableOwnerSlides:tableOwners,requiredNativeChartOwnerSlides:[11,15,16,19],materializeLiteralChartWorkbooks:true,fontPolicy:{basis:'reference',families:['Microsoft JhengHei','Consolas'],referencePath:reference,referenceSha256:createHash('sha256').update(await fs.readFile(reference)).digest('hex')},verifyArtifactToolImport:true,receiptPath:receipt});
console.log(JSON.stringify({event:'finalized',file:final,result}));
const p=await PresentationFile.importPptx(await FileBlob.load(final));
if(p.slides.items.length!==24)throw Error('Wrong final slide count');
for(let i=0;i<p.slides.items.length;i++){
  const s=p.slides.items[i],stem='slide-'+String(i+1).padStart(2,'0');
  const png=await p.export({slide:s,format:'png',scale:1.25});
  await fs.writeFile(path.join(renders,stem+'.png'),new Uint8Array(await png.arrayBuffer()),{flag:'wx'});
  const layout=await s.export({format:'layout'});
  await fs.writeFile(path.join(renders,stem+'.json'),await layout.text(),{flag:'wx'});
  console.log(JSON.stringify({event:'rendered',slide:i+1}));
}
console.log(JSON.stringify({status:'finalized_and_rendered_requires_visual_review',file:final,renders,slideCount:24,receipt}));
