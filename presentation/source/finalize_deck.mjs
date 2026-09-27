import fs from 'node:fs/promises';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { FileBlob, PresentationFile } from '@oai/artifact-tool';

process.env.RUNTIME_NODE_MODULES ||= 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';
const WORKSPACE = 'C:/NYCU_ML';
const TASK = path.join(WORKSPACE, 'outputs/presentation_v22_20260928');
const BUILD = path.join(TASK, 'build');
const DELIVERABLES = path.join(TASK, 'deliverables');
const SKILL = 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.905.11957/skills/presentations';
const PYTHON = 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe';
const CANDIDATE = path.join(BUILD, 'candidate.pptx');
const FINAL = path.join(DELIVERABLES, '臺灣房價預測_正式報告_v22.pptx');
const RENDERS = path.join(BUILD, 'final_renders');
const TABLE_OWNERS = [3, 4, 5, 7, 10, 15];
const CHART_OWNERS = [2, 6, 8, 9];

const { finalizePresentation } = await import(pathToFileURL(
  path.join(SKILL, 'container_tools/artifact_tool_utils.mjs'),
).href);
await fs.mkdir(DELIVERABLES, { recursive: true });
await fs.mkdir(RENDERS, { recursive: true });

const result = await finalizePresentation({
  workspaceDir: TASK,
  explicitTotalSlideCount: 16,
  candidatePath: CANDIDATE,
  finalPath: FINAL,
  pythonExecutable: PYTHON,
  integrityValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: [
    '--expected-slide-size-emu', '12192000,6858000',
    '--validate-bullet-geometry',
    '--validate-heading-fit',
    ...TABLE_OWNERS.flatMap(number => ['--require-native-table-slide', String(number)]),
  ],
  requiredNativeTableOwnerSlides: TABLE_OWNERS,
  requiredNativeChartOwnerSlides: CHART_OWNERS,
  // Newly authored charts contain complete literal snapshots of saved data.
  materializeLiteralChartWorkbooks: true,
  fontPolicy: { basis: 'design', families: ['Microsoft JhengHei', 'Consolas'] },
  verifyArtifactToolImport: true,
  receiptPath: path.join(BUILD, 'final.validation.json'),
});
console.log('FINALIZED', JSON.stringify(result));

const presentation = await PresentationFile.importPptx(await FileBlob.load(FINAL));
for (let index = 0; index < presentation.slides.items.length; index += 1) {
  const slide = presentation.slides.items[index];
  const stem = 'slide-' + String(index + 1).padStart(2, '0');
  const png = await presentation.export({ slide, format: 'png', scale: 1.25 });
  await fs.writeFile(path.join(RENDERS, stem + '.png'), new Uint8Array(await png.arrayBuffer()));
  const layout = await slide.export({ format: 'layout' });
  await fs.writeFile(path.join(RENDERS, stem + '.json'), await layout.text(), 'utf8');
  console.log('FINAL_RENDERED', index + 1);
}
console.log(JSON.stringify({ finalPath: FINAL, slides: presentation.slides.items.length, renders: RENDERS }));
