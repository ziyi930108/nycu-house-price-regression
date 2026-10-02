import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FileBlob, PresentationFile} from '@oai/artifact-tool';

const work = path.resolve(process.env.REVIEW_WORK_DIR);
const sources = path.resolve(process.env.REVIEW_SOURCE_DIR);
const key = process.argv[2];
const plan = JSON.parse(await fs.readFile(path.join(work, 'content/notes_plan.json'), 'utf8'));
const entry = plan.styles.find(x => x.style === key);
if (!entry || !Array.isArray(entry.changed_slide_numbers)) throw Error('Invalid style or notes plan');
const source = path.join(sources, entry.source_file);
const build = path.join(work, 'build', key);
if (createHash('sha256').update(await fs.readFile(source)).digest('hex') !== entry.source_sha256) throw Error('Source changed');
const p = await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot = await p.inspect({kind: 'slide,layout', maxChars: 200000});
await fs.writeFile(path.join(build, 'before.ndjson'), snapshot.ndjson, {flag: 'wx'});
const anchors = snapshot.ndjson.split('\n').filter(Boolean).map(x => JSON.parse(x)).filter(x => x.kind === 'slide');
if (anchors.length !== 23 || anchors.map(x => x.slide).sort((a,b) => a-b).join(',') !== Array.from({length:23}, (_,i) => i+1).join(',')) throw Error('Expected all 23 slide anchors');
const selected = new Set(entry.changed_slide_numbers);
for (const anchor of anchors) {
  if (selected.has(anchor.slide)) p.resolve(anchor.id).speakerNotes.textFrame.setText(entry.notes[anchor.slide - 1]);
}
const candidate = path.join(build, 'candidate.pptx');
try { await fs.access(candidate); throw Error('Candidate already exists'); } catch (e) { if (e.code !== 'ENOENT') throw e; }
await (await PresentationFile.exportPptx(p)).save(candidate);
console.log(JSON.stringify({style:key, candidate, changed_slide_numbers:[...selected]}));
