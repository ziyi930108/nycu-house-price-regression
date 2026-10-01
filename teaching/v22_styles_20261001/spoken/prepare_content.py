"""Integrate the user-requested spoken rewrite while preserving all slide content."""
from pathlib import Path
from datetime import datetime,timezone
from xml.etree import ElementTree as E
from zipfile import ZipFile
import copy,hashlib,json,math,re,statistics
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OLD=ROOT/'outputs/v22_teaching_refinement_20261001';CONTENT=HERE/'content'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def js(p,d):write(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def han(s):return len(re.findall(r'[\u4e00-\u9fff]',s))
def mmss(s):return f'{s//60}:{s%60:02}'
old=json.loads((OLD/'content/narrative.json').read_text(encoding='utf-8'))
proposal=json.loads((CONTENT/'approved_scripts.json').read_text(encoding='utf-8'))
assert [x['number'] for x in proposal['slides']]==list(range(1,21))
new=copy.deepcopy(old)
for target,source in zip(new['slides'][:20],proposal['slides']):
 assert set(source)=={'number','script'} and target['number']==source['number']
 assert source['script'].strip() and source['script']!=target['script']
 target['script']=source['script'].strip()
 target['spoken_han_characters']=han(target['script'])
 target['seconds']=int(math.floor((target['spoken_han_characters']/285*60)/5+.5)*5)
main=new['slides'][:20];count=sum(s['spoken_han_characters'] for s in main);seconds=sum(s['seconds'] for s in main)
assert 4400<=count<=5000 and count/300>=12,(count,seconds)
for a,b in zip(old['slides'],new['slides']):
 if a['number']>20:assert a==b
 else:assert {k:v for k,v in a.items() if k not in ('script','spoken_han_characters','seconds')}=={k:v for k,v in b.items() if k not in ('script','spoken_han_characters','seconds')}
new['meta'].update({'edition':'v22_spoken_20261001','main_spoken_han_characters':count,'planned_main_seconds':seconds,
 'estimated_minutes_at_250':count/250,'estimated_minutes_at_275':count/275,'estimated_minutes_at_285':count/285,
 'estimated_minutes_at_300':count/300,'main_exceeds_12_minutes_at_300':count/300>=12,
 'content_status':'spoken_scripts_prepared_for_notes_update','user_requested_change':'讓講解更白話、容易上台說'})
js(CONTENT/'narrative.json',new)
lines=['# 臺灣房屋總價預測：口語講稿','','適用v22三款23頁簡報。只改主講口述，第21至23頁附錄保持。日期：2026-10-01。','',
 f'主講20頁共有 **{count:,}個純漢字**，按250至300字／分鐘估計 **{count/300:.2f}至{count/250:.2f}分鐘**。逐頁規劃 **{mmss(seconds)}**，尚未真人計時。英文、數字、標點、附錄、问答及刻意停頓不計入。',
 '', '講法：每次先說一個意思，再指向畫面或說數字例子。段落間可以換氣，配速仍須親自排練。', '',
 '| 頁次 | 主題 | 主講純漢字 | 建議秒數 | 累計 |','| --- | --- | ---: | ---: | --- |']
total=0
for s in main:
 total+=s['seconds'];lines.append(f"| {s['number']:02} | {s['title']} | {s['spoken_han_characters']} | {s['seconds']} | {mmss(total)} |")
for s in new['slides']:
 lines += ['',f"## 第{s['number']:02}頁　{s['title']}",'',('附錄，不計主講。' if s['is_appendix'] else '**先記住這句：** '+s['takeaway']),'',s['script'],'','**這一步的理由：** '+s['why'],'','證據位置：']
 lines += ['- `'+src+'`' for src in s['sources']]
write(CONTENT/'完整口語講稿.md','\n'.join(lines)+'\n')
qa=(OLD/'content/排練與問答.md').read_text(encoding='utf-8')
old_timing='主講完整稿目前有 **4,789 個純漢字**，以每分鐘300字估計 **15.96 分鐘**，以285字估計 **16.80 分鐘**，以250字估計 **19.16 分鐘**。逐頁配速合計 **16分45秒**。'
assert qa.count(old_timing)==1
qa=qa.replace(old_timing,f'主講口語稿目前有 **{count:,} 個純漢字**，以每分鐘300字估計 **{count/300:.2f} 分鐘**，以285字估計 **{count/285:.2f} 分鐘**，以250字估計 **{count/250:.2f} 分鐘**。逐頁配速合計 **{seconds//60}分{seconds%60:02}秒**。')
assert qa.split('## 常見問答',1)[1]==(OLD/'content/排練與問答.md').read_text(encoding='utf-8').split('## 常見問答',1)[1]
write(CONTENT/'排練與問答.md',qa)
cues=['# 上台口述提示','','先完整練第1至20頁，附錄與問答不算主講時間。不要只念投影片的短句；下面每頁的提醒是找回思路用，完整說明仍在口語講稿。','',
 '1. 一句只講一個重點，句尾停一下再換句。','2. 第3、10、13、15頁把教學算例說完，再講它為什麼有用。','3. 第17頁指向完整Public分數，口述約兩百萬元即可，再說明Private仍未知。',
 '4. 真正卡住時先說這頁要解決什麼問題，再回到例子；不用從整段第一句重來。','',
 '| 頁次 | 想不起下一句時，先回到這個意思 |','| --- | --- |']
for s in main:cues.append(f"| {s['number']:02} | {s['takeaway']} |")
cues+=['','配速是字數估計，不是已經完成的真人排練。若實際少於12分鐘，先檢查有沒有漏講方法理由與數字例子，不靠重複結論補時間。']
write(CONTENT/'上台口述提示.md','\n'.join(cues)+'\n')
styles=json.loads((OLD/'delivery_receipt.json').read_text(encoding='utf-8'))['styles'];plans=[]
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
for e in styles:
 p=OLD/e['path'];assert sha(p)==e['sha256'];notes=[]
 with ZipFile(p) as z:
  for n in range(1,24):
   parts=[x.text or '' for x in E.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml')).findall('.//a:t',ns)]
   body='\n'.join(parts);previous=old['slides'][n-1]['script'];assert previous in body
   updated=body.replace(previous,new['slides'][n-1]['script'],1) if n<=20 else body
   notes.append(updated)
 plans.append({'style':e['style'],'source_file':p.name,'source_sha256':sha(p),'notes':notes})
js(CONTENT/'notes_plan.json',{'source_narrative_sha256':sha(OLD/'content/narrative.json'),'narrative_sha256':sha(CONTENT/'narrative.json'),'styles':plans})
def sentence_stats(slides):
 lengths=[han(t) for s in slides for t in re.split(r'[。！？]',s['script']) if han(t)]
 return {'sentence_count':len(lengths),'mean_han_per_sentence':statistics.mean(lengths),'median_han_per_sentence':statistics.median(lengths),'max_han_per_sentence':max(lengths),'sentences_over_40_han':sum(n>40 for n in lengths)}
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'user_request':'讓講解更白話、容易上台說','changed_scripts':list(range(1,21)),
 'slide_titles_keypoints_sources_why_and_appendices_unchanged':True,'main_han':count,'planned_main_seconds':seconds,
 'estimated_minutes_at_300':count/300,'estimated_minutes_at_250':count/250,'human_rehearsal_measured':False,
 'old_sentence_stats':sentence_stats(old['slides'][:20]),'new_sentence_stats':sentence_stats(main),
 'measurement_scope':'Sentence length is a writing metric, not proof of oral comprehension or delivery duration.',
 'sixteen_qa_answers_unchanged':True,'source_narrative_sha256':sha(OLD/'content/narrative.json'),
 'new_narrative_sha256':sha(CONTENT/'narrative.json'),'approved_scripts_sha256':sha(CONTENT/'approved_scripts.json'),
 'new_ML_fits':0,'Private_score':None}
js(CONTENT/'revision_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
