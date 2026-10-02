from pathlib import Path
import json, shutil, hashlib, html
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OUT=ROOT/'deliverables/v22_審查資料夾_20261001';VIDEO=WORK/'video_final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def cp(p,q):
    q.parent.mkdir(parents=True,exist_ok=True)
    assert not q.exists(),q
    shutil.copy2(p,q);assert sha(p)==sha(q)
r=json.loads((VIDEO/'receipt.json').read_text('utf-8'));assert r['status']=='completed' and r['main_mp4']['duration_seconds']==900
names={'main.mp4':'v22_15分鐘講解_合成旁白.mp4','captions.srt':'v22_15分鐘講解.srt','captions.vtt':'v22_15分鐘講解.vtt','chapters.json':'章節時間.json','receipt.json':'影片製作收據.json'}
for a,b in names.items():cp(VIDEO/a,OUT/'05_講解影片'/b)
for p in (WORK/'video_source').iterdir():
    if p.is_file() and p.suffix in ('.py','.ps1','.json','.md'):
        cp(p,OUT/'04_程式與證據/影片製作程式'/p.name)
cp(WORK/'build/execution.json',OUT/'04_程式與證據/本次簡報製作收據.json')
n=json.loads((WORK/'content/narrative.json').read_text('utf-8'))
chapters=json.loads((VIDEO/'chapters.json').read_text('utf-8'))
for x in chapters:x['title']=n['slides'][x['slide']-1]['title']
(OUT/'05_講解影片/章節時間.json').write_text(json.dumps(chapters,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
links=[]
for c in chapters:
    seconds=c['start_seconds'];stamp=f'{int(seconds)//60:02}:{int(seconds)%60:02}'
    links.append(f'<button type="button" class="secondary chapter" data-seek="{seconds:.3f}">{stamp}　{c["slide"]:02} {html.escape(c["title"])}</button>')
markup='''<p>白底學術版，20 頁主講，實際影片長度 <strong>15:00</strong>。使用 Windows 本機繁體中文合成旁白，並非真人錄音，也不能代替你自己的上台計時。</p><video controls preload="metadata" poster="assets/A-cover.png" aria-label="v22十五分鐘投影片講解"><source src="05_講解影片/v22_15分鐘講解_合成旁白.mp4" type="video/mp4">瀏覽器無法播放時，請下載MP4。</video><p><a href="05_講解影片/v22_15分鐘講解_合成旁白.mp4">開啟或下載完整 MP4</a>　<a href="05_講解影片/v22_15分鐘講解.srt">SRT 字幕</a></p><details><summary>依頁次跳到指定章節</summary><div style="display:flex;gap:8px;flex-wrap:wrap">'''+''.join(links)+'''</div></details><p class="meta">字幕已放在畫面下方，不遮住投影片。講稿全文保留，部分英文術語只作配音發音替換。未做真人逐句聽音評分。</p>'''
p=OUT/'index.html';s=p.read_text('utf-8');old='<div id="videoContent">影片會在簡報檢查完成後，依本版講稿製作。</div>';assert s.count(old)==1
s=s.replace(old,'<div id="videoContent">'+markup+'</div>')
s=s.replace('</script></body>',"document.querySelectorAll('.chapter').forEach(b=>b.addEventListener('click',()=>{const v=document.querySelector('video');v.currentTime=Number(b.dataset.seek);v.scrollIntoView({behavior:'smooth',block:'center'});v.play().catch(()=>{});}));</script></body>")
p.write_text(s,encoding='utf-8')
notes='''# 講解影片

先開啟MP4，或回到資料夾index.html的影片區。全片15分00秒，白底學術版20頁主講，附錄与問答不入片。字幕已燒入畫面下方，另附SRT/VTT。MP4內有20個章節，index可按頁次跳轉。

這是Windows本機Microsoft Hanhan Desktop繁體中文合成旁白，不是真人上台錄音。主稿共4,065漢字；Rate 0原音約19分12秒，因此改採Rate 2原音約15分25秒，保留全部講稿。段間停頓後，音訊以約1.052倍速度合成15分鐘影片。未截掉尾句，製作收據保留完整解碼與尾段靜音檢查。

模型名等英文作發音替換，字幕保留稿面文字，替換表在04_程式與證據/影片製作程式/speech_input.json。未連線付費配音服務，也未人工逐句聽音評分。影片可用來熟悉流程；你實際報告仍需自己計時。
'''
(OUT/'05_講解影片/先讀我.md').write_text(notes,encoding='utf-8')
p=OUT/'00_審查指南.md';s=p.read_text('utf-8');s+='\n\n## 講解影片已附上\n\n`05_講解影片`放有15分鐘MP4、字幕及章節。可先看一次建立流程概念，再練自己的講法。旁白是合成語音，影片時長不代表你已完成真人排練。\n';p.write_text(s,encoding='utf-8')
print(json.dumps({'video_integrated':True,'review_folder':str(OUT),'duration_seconds':900},ensure_ascii=False))
