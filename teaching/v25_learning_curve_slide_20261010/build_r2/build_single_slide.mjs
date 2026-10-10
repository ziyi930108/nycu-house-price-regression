/**
 * Source-only prepared on 2026-10-10. No numerical curve is embedded.
 * The caller must first run the Presentations operation marker exactly once.
 * Run from this private build directory, with node_modules linked to the
 * configured bundled Node packages. This program never fits a model.
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import crypto from 'node:crypto';
import {pathToFileURL, fileURLToPath} from 'node:url';
import {Presentation, PresentationFile} from '@oai/artifact-tool';

import {addLearningCurveSlide, validateCurveData} from './curve_slide.mjs';
export {addLearningCurveSlide} from './curve_slide.mjs';

process.env.RUNTIME_NODE_MODULES ??= 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules';

async function main() {
const args = process.argv.slice(2);
const options = {};
for (let i = 0; i < args.length; i += 2) {
  if (!args[i].startsWith('--') || !args[i + 1] || options[args[i]]) {
    throw new Error('Use unique --name value arguments.');
  }
  options[args[i]] = args[i + 1];
}
const allowed = new Set(['--curve', '--curve-sha', '--out', '--workspace-dir', '--skill-dir', '--runtime-python']);
for (const key of Object.keys(options)) if (!allowed.has(key)) throw new Error('Unknown argument: ' + key);
for (const key of ['--curve', '--curve-sha', '--out']) if (!options[key]) throw new Error('Required: ' + key);
const ROOT = path.resolve(options['--workspace-dir'] || 'C:/NYCU_ML');
const SKILL = path.resolve(options['--skill-dir'] || 'C:/Users/ziyi9/.codex/plugins/cache/openai-primary-runtime/presentations/26.1007.11041/skills/presentations');
const PYTHON = path.resolve(options['--runtime-python'] || 'C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe');
const BUILD = path.resolve(import.meta.dirname);
const CURVE = path.resolve(options['--curve']);
const FINAL = path.resolve(options['--out']);
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const need = (value, message) => { if (!value) throw new Error(message); };
function inWorkspace(filename) {
  const rel = path.relative(ROOT, filename);
  need(rel && rel !== '..' && !rel.startsWith('..' + path.sep) && !path.isAbsolute(rel), 'Path leaves workspace: ' + filename);
  return rel.replaceAll(path.sep, '/');
}
for (const filename of [BUILD, CURVE, FINAL]) inWorkspace(filename);
need(path.extname(FINAL).toLowerCase() === '.pptx', 'Final output must be a PPTX.');
need(path.dirname(FINAL) !== BUILD, 'Final output must be separate from the private build directory.');
const expectedSha = options['--curve-sha'];
need(/^[0-9a-f]{64}$/.test(expectedSha), 'Expected curve SHA256 required.');
const curveBytes = await fs.readFile(CURVE);
need(sha(curveBytes) === expectedSha, 'Curve bytes differ from the reviewed data.');
const d = JSON.parse(curveBytes.toString('utf8').replace(/^\uFEFF/, ''));
const contract = validateCurveData(d);
for (const source of d.sources) {
  need(typeof source.path === 'string' && /^[0-9a-f]{64}$/.test(source.sha256), 'Invalid curve source record.');
  const filename = path.resolve(ROOT, source.path);
  inWorkspace(filename);
  need(sha(await fs.readFile(filename)) === source.sha256, 'Curve source bytes differ: ' + source.path);
}

const DRAFT = path.join(BUILD, 'candidate.pptx');
const PREVIEW = path.join(BUILD, 'slide-01.png');
const LAYOUT = path.join(BUILD, 'presentation.json');
const RECEIPT = path.join(BUILD, 'single_slide.validation.json');
const DATA = path.join(BUILD, 'chart_source_data.json');
const SCRIPT = path.join(BUILD, 'speaker_script.md');
for (const filename of [FINAL, DRAFT, PREVIEW, LAYOUT, RECEIPT, DATA, SCRIPT]) {
  try { await fs.access(filename); throw new Error('Preserve existing output: ' + filename); }
  catch (error) { if (error.code !== 'ENOENT') throw error; }
}

const {finalizePresentation} = await import(pathToFileURL(path.join(SKILL, 'container_tools/artifact_tool_utils.mjs')).href);
const REFERENCE_FONT = 'Microsoft JhengHei';
const p = Presentation.create({slideSize: {width: 1280, height: 720}});
const {slide: s, spoken} = await addLearningCurveSlide(p, d, {skillDir: SKILL, fontFamily: REFERENCE_FONT, curveSource: {path: inWorkspace(CURVE), sha256: expectedSha}});
await fs.mkdir(path.dirname(FINAL), {recursive: true});
await fs.writeFile(DATA, JSON.stringify({curve: {path: inWorkspace(CURVE), sha256: expectedSha}, contract, epochs: d.epochs, train_mae_twd: d.train_mae_twd, validation_mae_twd: d.validation_mae_twd, chart_display_unit: '10000 TWD', chart_display_rounding: {decimal_places: 4, quantum_TWD: 1, method: 'Number((TWD / 10000).toFixed(4))', raw_curve_unchanged: true}, chart_display_values: {train_mae_10000_twd: d.train_mae_twd.map(v => Number((v / 10000).toFixed(4))), validation_mae_10000_twd: d.validation_mae_twd.map(v => Number((v / 10000).toFixed(4)))}, final_epoch_diagnostic: {train_mae_twd: d.train_mae_twd[29], validation_mae_twd: d.validation_mae_twd[29]}, sources: d.sources}, null, 2) + '\n', {flag: 'wx'});
await fs.writeFile(SCRIPT, '# 單頁補充講稿\n\n' + spoken + '\n\n此稿尚未本人計時，插入整套簡報後需重新核對約15分鐘的主講總長。\n', {flag: 'wx'});
await (await PresentationFile.exportPptx(p)).save(DRAFT);
await fs.writeFile(LAYOUT, JSON.stringify(p.toProto()), {flag: 'wx'});
const result = await finalizePresentation({
  workspaceDir: ROOT, candidatePath: DRAFT, finalPath: FINAL,
  pythonExecutable: PYTHON,
  integrityValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_package_integrity.py'),
  layoutValidatorPath: path.join(SKILL, 'container_tools/inspect_presentation_layout_geometry.py'),
  layoutArgs: ['--expected-slide-size-emu', '12192000,6858000', '--validate-heading-fit'],
  explicitTotalSlideCount: 1, requiredNativeTableOwnerSlides: [], requiredNativeChartOwnerSlides: [1],
  materializeLiteralChartWorkbooks: true,
  fontPolicy: {basis: 'reference', families: [REFERENCE_FONT], referencePath: path.join(ROOT, 'deliverables/v22_github/teaching/v22_styles_20261001/deliverables/臺灣房價預測_v22_白底學術_r2.pptx'), referenceSha256: sha(await fs.readFile(path.join(ROOT, 'deliverables/v22_github/teaching/v22_styles_20261001/deliverables/臺灣房價預測_v22_白底學術_r2.pptx')))},
  verifyArtifactToolImport: true, receiptPath: RECEIPT,
});
// Rendering is a QA artifact, not a replacement for the editable chart.
const png = await p.export({slide: s, format: 'png', scale: 1.25});
await fs.writeFile(PREVIEW, new Uint8Array(await png.arrayBuffer()), {flag: 'wx'});
console.log(JSON.stringify({status: 'built_requires_visual_review', final: FINAL, preview: PREVIEW, curve_sha256: expectedSha, editable_charts: 1, points_per_series: 30, new_model_fits: 0, finalizer: result}));

}
if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  await main();
}
