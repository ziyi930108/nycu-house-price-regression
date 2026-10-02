from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
import copy, hashlib, json, math, re, unicodedata

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'outputs/v22_15min_20261001'
CONTENT=HERE/'content'
CONTENT.mkdir(exist_ok=True)
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):
    with p.open('x',encoding='utf-8') as f: f.write(s)
def save(p,d): write(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
def han(s): return len(re.findall(r'[\u4e00-\u9fff]',s))
def clock(s): return f'{s//60}:{s%60:02}'
proposal=json.loads((HERE/'script_proposal.json').read_text('utf-8'))
assert sha(OLD/'content/narrative.json')==proposal['source_narrative_sha256']
old=json.loads((OLD/'content/narrative.json').read_text('utf-8'))
new=copy.deepcopy(old)
changes=[]
for c in proposal['changes']:
    page=c['page']; target=new['slides'][page-1]
    assert target['script']==c['before'] and page in (12,13)
    after=c['after'].replace('；','。')
    target['script']=after
    target['spoken_han_characters']=han(after)
    target['seconds']=int(math.floor((han(after)/275*60)/5+.5)*5)
    changes.append({**c,'after':after,'decision':'adopted; semicolons changed to full stops for oral sentences'})
main=new['slides'][:20]
count=sum(han(s['script']) for s in main)
seconds=sum(s['seconds'] for s in main)
assert count==4065 and seconds==885
for a,b in zip(old['slides'],new['slides']):
    if a['number'] not in (12,13): assert a==b
    else: assert {k:v for k,v in a.items() if k not in ('script','spoken_han_characters','seconds')}=={k:v for k,v in b.items() if k not in ('script','spoken_han_characters','seconds')}
new['meta'].update(edition='v22_review_20261001',main_spoken_han_characters=count,planned_main_seconds=seconds,content_status='approved_for_review_edition',user_requested_change='繼續優化並整理審查資料夾，再製作合成旁白影片')
for rate in (250,275,285,300): new['meta'][f'estimated_minutes_at_{rate}']=count/rate
save(CONTENT/'narrative.json',new)
save(CONTENT/'approved_changes.json',{'changed_pages':[12,13],'changes':changes,'base_sha256':sha(OLD/'content/narrative.json'),'narrative_sha256':sha(CONTENT/'narrative.json'),'visible_slides_unchanged':True,'human_rehearsal_measured':False,'private_score':None})
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
styles=[]
receipt=json.loads((OLD/'delivery_receipt.json').read_text('utf-8'))
for style in receipt['styles']:
    p=OLD/style['path']; assert sha(p)==style['sha256']
    notes=[]
    with ZipFile(p) as z:
        for n in range(1,24):
            body='\n'.join(t.text or '' for t in E.fromstring(z.read(f'ppt/notesSlides/notesSlide{n}.xml')).findall('.//a:t',ns))
            assert body.count(old['slides'][n-1]['script'])==1
            notes.append(body.replace(old['slides'][n-1]['script'],new['slides'][n-1]['script'],1))
    styles.append({'style':style['style'],'source_file':p.name,'source_sha256':sha(p),'notes':notes,'changed_slide_numbers':[12,13]})
save(CONTENT/'notes_plan.json',{'narrative_sha256':sha(CONTENT/'narrative.json'),'styles':styles})
lines=['# 臺灣房價預測：審查版完整口語講稿','','三款簡報共用。主講第1至20頁，附錄第21至23頁不計時。', '',f'全文{count:,}純漢字。按每分鐘275字估計{count/275:.2f}分鐘，逐頁規劃{clock(seconds)}。英文名稱、停頓和指圖時間需另外考慮，尚未本人計時。', '', '|頁次|主題|秒數參考|累計|','|---|---|---:|---:|']
t=0
for s in main:
    t+=s['seconds'];lines.append(f"|{s['number']}|{s['title']}|{s['seconds']}|{clock(t)}|")
for s in new['slides']:
    lines.extend(['',f"## 第{s['number']:02}頁　{s['title']}",'','附錄，不計主講。' if s['is_appendix'] else '**先記住：** '+s['takeaway'],'',s['script'],'','**設計理由：** '+s['why']])
write(CONTENT/'完整口語講稿.md','\n'.join(lines)+'\n')
qa=(OLD/'content/排練與問答.md').read_text('utf-8').replace('4,051','4,065').replace('13.50','13.55').replace('14.21','14.26').replace('16.20','16.26')
write(CONTENT/'排練與問答.md',qa)
write(CONTENT/'上台口述提示.md',(OLD/'content/上台口述提示.md').read_text('utf-8'))
template=(OLD/'rehearsal_template.html').read_text('utf-8').replace('v22 約15分鐘上台排練','v22 審查版上台排練').replace('原講稿本頁配速','本版講稿配速')
payload={'source_sha256':sha(CONTENT/'narrative.json'),'slides':[{k:s[k] for k in ('number','title','takeaway','script','seconds')} for s in main]}
assert template.count('__DECK_JSON__')==1
write(CONTENT/'rehearsal_template.html',template)
write(CONTENT/'v22-review-rehearsal.html',template.replace('__DECK_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
latin=re.findall(r'[A-Za-z]+',unicodedata.normalize('NFKC','\n'.join(s['script'] for s in main)))
save(CONTENT/'timing.json',{'main_han':count,'latin_labels':latin,'planned_seconds':seconds,'estimate_minutes_at_275':count/275,'range_minutes_at_250_to_300':[count/300,count/250],'human_rehearsal_measured':False,'video_timing_must_be_measured_separately':True})
print(json.dumps({'changed_pages':[12,13],'main_han':count,'planned_seconds':seconds,'latin_labels':len(latin)}))
