import fs from 'node:fs/promises';
import path from 'node:path';
import {createHash} from 'node:crypto';
import {FileBlob,PresentationFile} from '@oai/artifact-tool';
const DIR=path.resolve(import.meta.dirname,'..');
const OLD=path.resolve(DIR,'../v22_teaching_refinement_20261001');
const plan=JSON.parse(await fs.readFile(path.join(DIR,'content/notes_plan.json'),'utf8'));
const key=process.argv[2];const entry=plan.styles.find(x=>x.style===key);if(!entry)throw Error('Expected A, B or C');
const build=path.join(DIR,'build',key);await fs.mkdir(build,{recursive:true});
const source=path.join(OLD,'deliverables',entry.source_file);
if(createHash('sha256').update(await fs.readFile(source)).digest('hex')!==entry.source_sha256)throw Error('Source changed');
const p=await PresentationFile.importPptx(await FileBlob.load(source));
const snapshot=await p.inspect({kind:'slide,layout',maxChars:200000});
await fs.writeFile(path.join(build,'before.ndjson'),snapshot.ndjson,{flag:'wx'});
const anchors=snapshot.ndjson.split('\n').filter(Boolean).map(x=>JSON.parse(x)).filter(x=>x.kind==='slide');
if(anchors.length!==23)throw Error('Expected 23 slides');
for(const a of anchors){if(a.slide<=20)p.resolve(a.id).speakerNotes.textFrame.setText(entry.notes[a.slide-1]);}
const candidate=path.join(build,'candidate.pptx');
try{await fs.access(candidate);throw Error('Candidate exists');}catch(e){if(e.code!=='ENOENT')throw e;}
await (await PresentationFile.exportPptx(p)).save(candidate);
console.log(JSON.stringify({style:key,candidate,editedNotes:20}));
