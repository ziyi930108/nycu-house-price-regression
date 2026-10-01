"""Copy this completed edition and update navigation without replacing historical artifacts."""
from pathlib import Path
from datetime import datetime,timezone
import json,hashlib,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];REPO=ROOT/'deliverables/v22_github'
PUB=ROOT/'outputs/github_publish_v22_15min_20261001';PREV=ROOT/'outputs/github_publish_v22_rehearsal_tool_20261001'
DEST=REPO/'teaching/v22_styles_20261001/fifteen_minute'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
assert json.loads((HERE/'browser_verification.json').read_text(encoding='utf-8'))['passed']
assert not PUB.exists() and not DEST.exists()
PUB.mkdir();DEST.mkdir()
shutil.copy2(REPO/'SOURCE_MANIFEST.json',PUB/'source_manifest_before.json')
shutil.copy2(PREV/'protected_before.json',PUB/'protected_before.json')
before={e['path']:e['sha256'] for e in json.loads((REPO/'SOURCE_MANIFEST.json').read_text(encoding='utf-8'))['files']}
protected=json.loads((PUB/'protected_before.json').read_text(encoding='utf-8'));assert all(sha(ROOT/p)==h for p,h in protected.items())
save(HERE/'review_and_attempts.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'root_review':'All 20 scripts read; desktop 1440px and mobile 390px page 12 expanded screenshots inspected without clipped or overlapping text. The 69 slide render hashes match r2, not a claim of new manual review of all slides.','agent_review':'stage_bundle_review verified exact template replacements and both updated Markdown timing sections; no stale 17-minute wording. Suggested optional label change from original-script pace to this-edition pace was deferred; actual displayed seconds use the new narrative.','browser_attempt1':'Test harness changed simulated state but left results hidden, so download event timed out after 30 seconds. Original test source preserved. Corrected harness reveals results; no app change.','git_read_attempt':'Default sandbox identity caused dubious ownership; rerun read-only as repository owner succeeded without modifying safe.directory configuration.','human_rehearsal_measured':False})
files=[]
for pat in ('*.py','content/*.json','content/*.md','build/*.mjs','*.json','*.png','rehearsal_template.html','verify_browser.cjs','verify_browser_attempt1.cjs','*.log','deliverables/*.pptx','deliverables/*.md','deliverables/*.html'):
 files+=list(HERE.glob(pat))
copied=[]
for p in sorted(set(files)):
 rel=p.relative_to(HERE);q=DEST/rel;q.parent.mkdir(parents=True,exist_ok=True);assert not q.exists();shutil.copy2(p,q)
 assert sha(p)==sha(q);copied.append({'source':p.relative_to(ROOT).as_posix(),'destination':q.relative_to(REPO).as_posix(),'sha256':sha(q)})
readme='''# v22 約15分鐘上台版

使用者最新要求「15分鐘上下，不要超過太多或是太少」。本版20頁主講共4,051純漢字，比舊口語版減少930字，逐頁規劃14分45秒。按275字／分估計14.73分鐘；250–300字／分為13.50–16.20分鐘，尚未本人計時。至少12分鐘，不靠附錄、問答或等待補時間。

| 風格 | 新PowerPoint | 相同畫面的PDF備份 |
| --- | --- | --- |
| 白底學術 | [簡報](deliverables/臺灣房價預測_v22_白底學術_15分鐘版.pptx) | [PDF](../projection/臺灣房價預測_v22_白底學術_投影備份.pdf) |
| 深色科技 | [簡報](deliverables/臺灣房價預測_v22_深色科技_15分鐘版.pptx) | [PDF](../projection/臺灣房價預測_v22_深色科技_投影備份.pdf) |
| 暖色雜誌 | [簡報](deliverables/臺灣房價預測_v22_暖色雜誌_15分鐘版.pptx) | [PDF](../projection/臺灣房價預測_v22_暖色雜誌_投影備份.pdf) |

三款都是同一個v22，選一款報告即可。20頁主講notes更新，3頁附錄保持。69張渲染與原r2相同、其餘OPC parts與圖表內嵌工作表保持原bytes。PDF沒有講者備忘稿。

先讀[完整口語講稿](content/完整口語講稿.md)和[每頁提示](content/上台口述提示.md)，再下載[離線排練HTML](deliverables/v22-15min-rehearsal.html)到電腦用瀏覽器開啟。GitHub原始碼頁不直接執行HTML。新版紀錄鍵與舊稿分開，20頁新稿與885秒計畫已核對。14–16分鐘只是工具的練習提示範圍，不是老師另定的硬門檻。工具沒有錄音或自動判斷內容完整。

[16題問答](content/排練與問答.md)與[四頁答辯卡](../spoken/defense/README.md)供補充，不算主講。[交付收據](delivery_receipt.json)、[文字核准](content/editor_review.json)、[瀏覽器檢查](browser_verification.json)和[審查及測試修正](review_and_attempts.json)說明驗證範圍。瀏覽器檢查使用模擬狀態，不能當成本人排練；尚未桌面PowerPoint或現場試播。

來源保存供追溯，這批prepare/build/verify腳本依原C:/NYCU_ML目錄與既有runtime，不是可直接從clone執行的新通用入口。既有scripts/rebuild_spoken_teaching.py仍重建舊4,981字口語版，不能宣稱已重建本版。完整13檔上台ZIP保留在原工作區outputs/v22_15min_20261001，未加入repo；repo已提供三款簡報、新稿、HTML及既有PDF/答辯卡連結。

Public MAE仍為1,998,568.5706元，Private未知。沒有新模型訓練、CSV或Kaggle操作，R106研究保持停止。此為文件與程式快照，不是完整重訓備份。
'''
(DEST/'README.md').write_text(readme,encoding='utf-8')
intro='最新[約15分鐘上台版](teaching/v22_styles_20261001/fifteen_minute/README.md)已完成三款PPTX、4,051字白話稿及離線排練工具，主講規劃14:45，尚未本人計時。原模型v22、Public及未知Private保持。\n\n'
for name in ('README.md','HANDOFF.md'):
 p=REPO/name;s=p.read_text(encoding='utf-8');first,body=s.split('\n\n',1)
 body=body.replace('最新[白話口語講稿版]','先前[白話口語講稿版]').replace('[最新口語版三款與講稿]','[先前口語版三款與講稿]').replace('更新日期：2026-10-01。最新[口語版]','前次更新：2026-10-01。[舊口語版]')
 if name=='HANDOFF.md':
  body=body.replace('| 簡報 | teaching/v22_styles_20261001/spoken內三款23頁口語版、4,981字講稿與提示，主講規劃1,040秒；舊版保留，未真人計時 |','| 簡報 | fifteen_minute內三款23頁簡報、4,051字口語稿與排練HTML，主講規劃885秒，未真人計時；舊版保持 |')
  body=body.replace('本頁於發布前核驗，base 69b556382278c69464ead99c7b9cbfc83ad55744','本頁於發布前核驗，base c27f92318fec556bc54b5c7e4ce1a97fd7e04d48')
 p.write_text(first+'\n\n'+intro+body,encoding='utf-8')
p=REPO/'teaching/v22_styles_20261001/README.md';s=p.read_text(encoding='utf-8');s=s.replace('最新主講請用[白話口語版](spoken/README.md)：20頁4,981字、規劃17:20（未真人計時）。','最新主講請用[約15分鐘版](fifteen_minute/README.md)：20頁4,051字、規劃14:45（未真人計時）。[舊口語版](spoken/README.md)保留。');p.write_text(s,encoding='utf-8')
p=REPO/'teaching/v22_styles_20261001/spoken/README.md';s=p.read_text(encoding='utf-8');head,body=s.split('\n\n',1);p.write_text(head+'\n\n本頁是舊17:20口語版。最新使用者要求約15分鐘，請先用[新上台版](../fifteen_minute/README.md)。此處來源及重建紀錄保留原身分。\n\n'+body,encoding='utf-8')
p=REPO/'teaching/v22_styles_20261001/projection/README.md';s=p.read_text(encoding='utf-8');s=s.replace('使用者尚未選定風格；r2代表簡報修訂，模型仍為v22。','使用者尚未選定風格；最新主講用約15分鐘稿，模型仍為v22。')
for style in ('白底學術','深色科技','暖色雜誌'):s=s.replace('../deliverables/臺灣房價預測_v22_'+style+'_r2.pptx','../fifteen_minute/deliverables/臺灣房價預測_v22_'+style+'_15分鐘版.pptx')
start=s.index('[完整講稿]');end=s.index('\n\n',start)
s=s[:start]+'[最新完整口語稿](../fifteen_minute/content/完整口語講稿.md)主講4,051純漢字，規劃14:45，按275字／分估計14.73分鐘；250–300字／分約13.50–16.20分鐘。目標約15分鐘、至少12分鐘，尚未本人計時。用[新排練工具](../fifteen_minute/README.md)自然說一遍，附錄與問答不計主講。'+s[end:];p.write_text(s,encoding='utf-8')
p=REPO/'AGENTS.md';s=p.read_text(encoding='utf-8');s+='\n2026-10-01最新：主講目標15分鐘上下，至少12分鐘，fifteen_minute為新稿入口。額度門檻依最新人類要求改為觀察到0%停止新增工作，不購買、不自行重置、不啟用點數；唯讀額度不能強制帳號扣點硬上限。每階段查額度並保存交接。\n';p.write_text(s,encoding='utf-8')
p=REPO/'scripts/package_source.py';s=p.read_text(encoding='utf-8');needle='\ndef sha(p):';assert needle in s
extra='\nTEACHING_EXTRA.update({"teaching/v22_styles_20261001/fifteen_minute/"+name for name in ("deliverables/v22-15min-rehearsal.html","rehearsal_template.html","verify_browser.cjs","verify_browser_attempt1.cjs")})\n'
s=s.replace(needle,extra+needle,1);p.write_text(s,encoding='utf-8')
# Logs are preserved in the original workspace; source package suffix policy excludes them.
added=[p.relative_to(REPO).as_posix() for p in DEST.rglob('*') if p.is_file() and p.suffix!='.log']
changed=sorted(p for p,h in before.items() if sha(REPO/p)!=h)
save(PUB/'source_copy_manifest.json',{'added_paths':sorted(added),'changed_existing':changed,'copies':copied,'portable_rebuild_for_this_edition_verified':False})
pub=(PREV/'publish.py').read_text(encoding='utf-8').replace('EXPECTED="f2b96edeadbfc49425554ab8f32b6cf0d8b68531"','EXPECTED="c27f92318fec556bc54b5c7e4ce1a97fd7e04d48"').replace('Add offline rehearsal with approved scripts and verified timing controls','Add 15-minute v22 talk scripts and synchronized presentation notes')
(PUB/'publish.py').write_text(pub,encoding='utf-8');shutil.copy2(PREV/'launch.py',PUB/'launch.py')
package=(PREV/'package_and_verify.py').read_text(encoding='utf-8').replace("v22_github_release_20261001_rehearsal_tool.zip","v22_github_release_20261001_15min.zip")
package=package.replace("for name in ['README.md'","for name in ['teaching/v22_styles_20261001/fifteen_minute/README.md','README.md'")
(PUB/'package_and_verify.py').write_text(package,encoding='utf-8')
print(json.dumps({'copied_files':len(copied),'added_source_files':len(added),'changed_existing':changed},ensure_ascii=False))
