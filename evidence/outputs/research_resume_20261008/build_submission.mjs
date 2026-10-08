import fs from 'node:fs/promises';
import crypto from 'node:crypto';
import {Workbook} from '@oai/artifact-tool';

const workbook=Workbook.create();
if(process.argv.includes('--help-csv')) {
  console.log(workbook.help('*',{search:'toCSV|exportCsv|exportCSV|csv',include:'index,examples,notes',maxChars:5000}).ndjson);
} else {
  const [inputPath,outputPath,evidenceDir]=process.argv.slice(2);
  if(!inputPath || !outputPath || !evidenceDir) throw new Error('input JSON, new CSV path, evidence directory required');
  const inputBytes=await fs.readFile(inputPath);
  const input=JSON.parse(inputBytes.toString('utf8'));
  if(input.status!=='full_model_prediction_sealed' || input.rows.length!==47073) throw new Error('Unsealed or incorrect rows');
  const seen=new Set();
  for(const [id,price] of input.rows) {
    if(typeof id!=='string' || !/^\d+$/.test(id) || seen.has(id)) throw new Error('Invalid/duplicate ID');
    seen.add(id);
    if(!Number.isSafeInteger(price) || price<1) throw new Error('Price is not a positive safe integer');
  }
  const sheet=workbook.worksheets.add('Submission');
  const rows=[['編號','總價元'],...input.rows];
  sheet.getRange(`A1:B${rows.length}`).values=rows;
  sheet.getRange('A1:B7').format.font={name:'Arial',size:12};
  sheet.getRange('A1:A7').format.columnWidthPx=160;
  sheet.getRange('B1:B7').format.columnWidthPx=220;
  sheet.getRange('A1:B1').format.font={name:'Arial',size:12,bold:true};
  sheet.getRange('B2:B7').format.numberFormat='0';
  workbook.recalculate();
  const written=sheet.getRange(`A1:B${rows.length}`).values;
  if(JSON.stringify(written)!==JSON.stringify(rows)) throw new Error('Artifact workbook altered values/order/types');
  const inspect=await workbook.inspect({kind:'table',range:'Submission!A1:B7',include:'values,formulas',tableMaxRows:7,tableMaxCols:2,maxChars:1800});
  const preview=await workbook.render({sheetName:'Submission',range:'A1:B7',scale:1.5,format:'png'});
  await fs.writeFile(`${evidenceDir}/csv_preview.png`,new Uint8Array(await preview.arrayBuffer()),{flag:'wx'});
  // No native CSV-export API is documented/discoverable in this runtime. Export
  // the artifact workbook's exact typed range through standard CSV encoding.
  // Allowed fields are digits or fixed Chinese headers, so quoting is unnecessary.
  const bytes=Buffer.from('\uFEFF'+written.map(row=>row.join(',')).join('\r\n')+'\r\n','utf8');
  await fs.writeFile(outputPath,bytes,{flag:'wx'});
  const receipt={status:'artifact_typed_range_exported_to_csv',rows:input.rows.length,columns:rows[0],
    input_json_sha256:crypto.createHash('sha256').update(inputBytes).digest('hex'),
    csv_sha256:crypto.createHash('sha256').update(bytes).digest('hex'),
    full_workbook_values_exact:true,formula_count:0,utf8_bom:true,
    csv_encoding:'standard comma-delimited CRLF from exact artifact range; native CSV API unavailable',
    preview:'csv_preview.png',inspection:inspect.ndjson};
  await fs.writeFile(`${evidenceDir}/csv_authoring_receipt.json`,JSON.stringify(receipt,null,2)+'\n',{flag:'wx'});
  console.log(JSON.stringify({rows:input.rows.length,csv:outputPath,sha256:receipt.csv_sha256}));
}
