// Tests use a simulated browser clock, never human rehearsal evidence.
const {chromium}=require('C:/Users/ziyi9/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/playwright');
const fs=require('node:fs'),path=require('node:path'),{pathToFileURL}=require('node:url'),assert=require('node:assert/strict');
(async()=>{
 const browser=await chromium.launch({headless:true,executablePath:'C:/Program Files (x86)/Microsoft/Edge/Application/msedge.exe'});
 try{
 const context=await browser.newContext({viewport:{width:1440,height:1000},colorScheme:'light'}),page=await context.newPage();
 const errors=[],requests=[];page.on('pageerror',e=>errors.push(e.message));page.on('request',r=>requests.push(r.url()));
 await page.clock.install();await page.goto(pathToFileURL(path.join(__dirname,'v22-rehearsal-reviewed.html')).href);
 const deck=JSON.parse(await page.locator('#deckData').textContent());assert.equal(deck.slides.length,20);
 for(const s of deck.slides){await page.locator('#pageSelect').selectOption(String(s.number-1));assert.equal(await page.locator('#script').textContent(),s.script);assert.equal(await page.locator('#cue').textContent(),s.takeaway);}
 await page.locator('#pageSelect').selectOption('0');assert(await page.locator('#prev').isDisabled());
 await page.locator('#toggle').click();await page.clock.fastForward(61000);await page.locator('#next').click();await page.clock.fastForward(20000);
 await page.locator('#toggle').click();const paused=await page.locator('#total').textContent();await page.clock.fastForward(600000);assert.equal(await page.locator('#total').textContent(),paused);
 await page.locator('#covered').check();await page.locator('#note').fill('測試用筆記：容易卡住');await page.locator('#toggle').click();await page.clock.fastForward(30000);await page.locator('#finish').click();
 assert.match(await page.locator('#total').textContent(),/^01:5[12]$/);assert.equal(await page.locator('#rows tr').count(),20);assert.match(await page.locator('#resultSummary').textContent(),/1 \/ 20/);
 await page.locator('#covered').uncheck();assert.match(await page.locator('#resultSummary').textContent(),/0 \/ 20/);
 await page.locator('#note').fill('更新的測試筆記');assert.match(await page.locator('#rows').textContent(),/更新的測試筆記/);
 await page.locator('#toggle').click();assert(await page.locator('#results').isHidden());
 await page.evaluate(()=>dispatchEvent(new PageTransitionEvent('pagehide',{persisted:true})));const away=await page.locator('#total').textContent();await page.clock.fastForward(600000);await page.evaluate(()=>dispatchEvent(new PageTransitionEvent('pageshow',{persisted:true})));assert.equal(await page.locator('#total').textContent(),away);
 await page.reload();assert.equal(await page.locator('#note').inputValue(),'更新的測試筆記');assert.equal(await page.locator('#toggle').textContent(),'繼續計時');const restored=await page.locator('#total').textContent();await page.clock.fastForward(60000);assert.equal(await page.locator('#total').textContent(),restored);
 await page.locator('#toggle').click();await page.clock.fastForward(620000);await page.locator('#finish').click();assert.match(await page.locator('#resultSummary').textContent(),/已到 12 分鐘/);assert.match(await page.locator('#resultSummary').textContent(),/0 \/ 20/);
 const downloadPromise=page.waitForEvent('download');await page.locator('#download').click();const download=await downloadPromise;const report=JSON.parse(fs.readFileSync(await download.path(),'utf8'));assert.equal(report.content_completion_verified,false);assert.equal(report.self_confirmed_pages,0);assert(report.total_seconds>=720);assert(report.slides[0].seconds>=61 && report.slides[0].seconds<62);assert.equal(report.slides.length,20);
 page.once('dialog',d=>d.dismiss());await page.locator('#reset').click();assert.notEqual(await page.locator('#total').textContent(),'00:00');
 page.once('dialog',d=>d.accept());await page.locator('#reset').click();assert.equal(await page.locator('#total').textContent(),'00:00');assert.equal(await page.locator('#pageSelect').inputValue(),'0');
 await page.locator('#pageSelect').selectOption('11');await page.locator('summary').click();await page.screenshot({path:path.join(__dirname,'desktop.png'),fullPage:true});
 await page.setViewportSize({width:390,height:844});const overflow=await page.evaluate(()=>document.documentElement.scrollWidth>innerWidth);assert.equal(overflow,false);await page.screenshot({path:path.join(__dirname,'mobile.png'),fullPage:true});
 assert.deepEqual(errors,[]);assert(requests.every(x=>x.startsWith('file:')));
 const receipt={passed:true,simulated_clock:true,human_rehearsal_completed:false,checks:['all 20 script and cue exact matches','page timing allocation','pause excluded','editable summary refreshed','resume hides stale summary','pagehide/pageshow simulation excludes away time','reload restores paused','12-minute threshold separate from completion','JSON download contents','reset cancel and accept','390px no horizontal overflow','no page errors or network resources'],test_report:report,requests};
 fs.writeFileSync(path.join(__dirname,'browser_verification.json'),JSON.stringify(receipt,null,2));console.log(JSON.stringify({passed:true,checks:receipt.checks.length,simulated_clock:true}));
 }finally{await browser.close();}
})().catch(e=>{console.error(e);process.exitCode=1;});
