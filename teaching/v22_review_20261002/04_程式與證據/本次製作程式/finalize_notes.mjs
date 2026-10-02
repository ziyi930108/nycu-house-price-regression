import fs from 'node:fs/promises';
import path from 'node:path';
import {pathToFileURL} from 'node:url';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const work = path.resolve(process.env.REVIEW_WORK_DIR);
const sources = path.resolve(process.env.REVIEW_SOURCE_DIR);
const skill = path.resolve(process.env.REVIEW_SKILL_DIR);
const python = process.env.RUNTIME_PYTHON;
const key = process.argv[2];
const plan = JSON.parse(await fs.readFile(path.join(work, 'content/notes_plan.json'), 'utf8'));
const entry = plan.styles.find(x => x.style === key);
if (!entry || !entry.source_file.endsWith('_15分鐘版.pptx')) throw Error('Unexpected source edition');
const build = path.join(work, 'build', key);
const source = path.join(sources, entry.source_file);
const final = path.join(work, 'deliverables', entry.source_file.replace('_15分鐘版.pptx', '_審查版.pptx'));
const renders = path.join(build, 'renders');
await fs.mkdir(renders);
const {finalizePresentation} = await import(pathToFileURL(path.join(skill, 'container_tools/artifact_tool_utils.mjs')).href);
const tableOwners = key === 'B' ? [4,6,7,8,10,12,13,22,23] : [3,4,6,7,8,10,12,13,22,23];
const chartOwners = key === 'B' ? [3,11,15,16] : [11,15,16];
await finalizePresentation({
  workspaceDir:work, explicitTotalSlideCount:23,
  candidatePath:path.join(build, 'candidate_preserved.pptx'), finalPath:final,
  pythonExecutable:python,
  integrityValidatorPath:path.join(skill, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath:path.join(skill, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs:['--expected-slide-size-emu','12192000,6858000','--validate-bullet-geometry','--validate-heading-fit',...tableOwners.flatMap(n => ['--require-native-table-slide',String(n)])],
  requiredNativeTableOwnerSlides:tableOwners,
  requiredNativeChartOwnerSlides:chartOwners,
  requiredEmbeddedWorkbookChartOwnerSlides:chartOwners,
  materializeLiteralChartWorkbooks:false,
  fontPolicy:{basis:'reference', families:key === 'C' ? ['Microsoft JhengHei','Consolas','PMingLiU'] : ['Microsoft JhengHei','Consolas'], referencePath:source, referenceSha256:entry.source_sha256},
  verifyArtifactToolImport:true, receiptPath:path.join(build, 'final.validation.json'),
});
const p = await PresentationFile.importPptx(await FileBlob.load(final));
if (p.slides.items.length !== 23) throw Error('Final slide count changed');
for (let i = 0; i < p.slides.items.length; i++) {
  const png = await p.export({slide:p.slides.items[i], format:'png', scale:1.25});
  await fs.writeFile(path.join(renders, `slide-${String(i+1).padStart(2,'0')}.png`), new Uint8Array(await png.arrayBuffer()), {flag:'wx'});
}
console.log(JSON.stringify({style:key,file:final,renders,slideCount:p.slides.items.length}));
