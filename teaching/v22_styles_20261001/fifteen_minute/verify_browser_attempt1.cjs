// Focused checks for revised content and timing guidance. No human rehearsal claim.
const {chromium}=require('C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),{pathToFileURL}=require('node:url'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000},colorScheme:'light'}),errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.clock.install();await page.goto(pathToFileURL(path.join(__dirname,'deliverables/v22-15min-rehearsal.html')).href);
  const deck=JSON.parse(await page.locator('#deckData').textContent()),n=JSON.parse(fs.readFileSync(path.join(__dirname,'content/narrative.json'),'utf8'));
  assert.equal(deck.slides.length,20);assert.equal(deck.slides.reduce((a,s)=>a+s.seconds,0),885);
  for(const [i,s] of deck.slides.entries()){
   assert.equal(s.script,n.slides[i].script);assert.equal(s.seconds,n.slides[i].seconds);
   await page.locator('#pageSelect').selectOption(String(i));assert.equal(await page.locator('#script').textContent(),s.script);
  }
  assert(!(await page.locator('body').textContent()).includes('17:20'));
  const cases=[[719,'尚未到12'],[720,'已滿12'],[839,'已滿12'],[840,'接近15'],[900,'接近15'],[960,'接近15'],[961,'超過16']];
  for(const [seconds,text] of cases){
   await page.evaluate(seconds=>{state.ms=Array(20).fill(0);state.ms[0]=seconds*1000;summary();},seconds);
   assert((await page.locator('#resultSummary').textContent()).includes(text));
  }
  await page.evaluate(()=>{state.ms=Array(20).fill(0);state.ms[0]=900000;summary();});
  const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();const download=await downloadPromise;
  const report=JSON.parse(fs.readFileSync(await download.path(),'utf8'));
  assert.equal(report.target_seconds,900);assert.equal(report.minimum_seconds,720);assert.equal(report.total_seconds,900);assert.equal(report.content_completion_verified,false);
  await page.locator('#pageSelect').selectOption('11');await page.locator('summary').click();await page.screenshot({path:path.join(__dirname,'desktop.png'),fullPage:true});
  await page.setViewportSize({width:390,height:844});assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:path.join(__dirname,'mobile.png'),fullPage:true});
  assert.deepEqual(errors,[]);assert(requests.every(x=>x.startsWith('file:')));
  const receipt={passed:true,scope:'Updated script exact matches, 885-second planning sum, all seven boundary cases, exported 900-second target, responsive layout and no network requests. Existing stopwatch logic unchanged; not an exhaustive repeat of prior tests.',boundary_cases:cases,human_rehearsal_measured:false,simulated_state:true,script_source_sha256:deck.source_sha256,requests};
  fs.writeFileSync(path.join(__dirname,'browser_verification.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
