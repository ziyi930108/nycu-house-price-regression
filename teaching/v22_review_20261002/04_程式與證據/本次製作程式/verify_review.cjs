const {chromium}=require('C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{pathToFileURL,fileURLToPath}=require('node:url');
(async()=>{
 const root=path.resolve(__dirname,'../../deliverables/v22_審查資料夾_20261001');
 const n=JSON.parse(fs.readFileSync(path.join(__dirname,'content/narrative.json'),'utf8'));
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
  const page=await browser.newPage({viewport:{width:1440,height:1000},colorScheme:'light'});const errors=[],requests=[];
  page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
  await page.goto(pathToFileURL(path.join(root,'index.html')).href);
  assert.equal(await page.locator('details.page').count(),23);
  for(const s of n.slides){assert.equal((await page.locator(`[data-script="${s.number}"]`).textContent()).replace(/\s/g,''),s.script.replace(/\s/g,''));}
  for(const href of await page.locator('a[href]').evaluateAll(a=>a.map(x=>x.href))){const u=new URL(href);if(u.protocol==='file:'&&!u.hash)assert(fs.existsSync(fileURLToPath(u)),href);}
  assert(await page.locator('.styles img').evaluateAll(imgs=>imgs.length===3&&imgs.every(x=>x.complete&&x.naturalWidth>0)));
  await page.screenshot({path:path.join(__dirname,'review-desktop.png')});
  await page.locator('#openAll').click();assert.equal(await page.locator('details.page[open]').count(),23);
  await page.locator('#closeAll').click();assert.equal(await page.locator('details.page[open]').count(),0);
  await page.locator('#style').selectOption({label:'白底學術'});await page.locator('#verdict').selectOption('revise');await page.locator('#minutes').fill('15.2');await page.locator('#pages').fill('12、13');await page.locator('#comments').fill('自動測試，並非使用者審查回饋。');
  const pending=page.waitForEvent('download');await page.locator('button[type=submit]').click();const dl=await pending;const feedback=JSON.parse(fs.readFileSync(await dl.path(),'utf8'));
  assert.equal(feedback.human_rehearsal_minutes,15.2);assert.equal(feedback.private_score,null);assert.equal(feedback.comments,'自動測試，並非使用者審查回饋。');
  await page.setViewportSize({width:390,height:844});await page.evaluate(()=>window.scrollTo(0,0));assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.screenshot({path:path.join(__dirname,'review-mobile.png')});
  await page.locator('#openAll').click();assert.equal(await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth),false);await page.locator('#closeAll').click();
  const video=page.locator('video');let videoTested=false;if(await video.count()){
   await video.evaluate(v=>{v.load();});await page.waitForFunction(()=>document.querySelector('video').readyState>=1);
   const media=await video.evaluate(v=>({duration:v.duration,width:v.videoWidth,height:v.videoHeight}));assert(Math.abs(media.duration-900)<1);assert.equal(media.width,1920);assert.equal(media.height,1080);
   await video.evaluate(v=>{v.muted=true;v.currentTime=590;});await video.evaluate(v=>v.play());await page.waitForFunction(()=>document.querySelector('video').currentTime>590);await video.evaluate(v=>v.pause());videoTested=true;
  }
  await page.goto(pathToFileURL(path.join(root,'02_講稿與排練/v22-review-rehearsal.html')).href);
  const data=JSON.parse(await page.locator('#deckData').textContent());assert.equal(data.slides.length,20);assert.equal(data.slides.reduce((a,s)=>a+s.seconds,0),885);
  for(let i=0;i<20;i++){await page.locator('#pageSelect').selectOption(String(i));assert.equal(await page.locator('#script').textContent(),n.slides[i].script);}
  assert.deepEqual(errors,[]);assert(requests.every(u=>u.startsWith('file:')||u.startsWith('blob:')));
  const receipt={passed:true,review_scripts_matched:23,rehearsal_scripts_matched:20,feedback_download_tested:true,feedback_is_synthetic_test_data:true,no_network_requests:true,mobile_no_horizontal_overflow:true,video_metadata_and_seek_tested:videoTested,human_rehearsal_measured:false,errors};
  fs.writeFileSync(path.join(__dirname,'browser_verification.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify(receipt));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
