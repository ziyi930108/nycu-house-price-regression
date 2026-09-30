import json,hashlib,re,zipfile
from pathlib import Path
from datetime import datetime,timezone
from lxml import etree
D=Path(__file__).resolve().parents[1]

def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 source=D/'content/narrative.json'
 n=json.loads(source.read_text(encoding='utf-8-sig'))
 slides=n['slides']; main=[s for s in slides if not s['is_appendix']]
 assert len(slides)==23 and len(main)==20
 assert [s['number'] for s in slides]==list(range(1,24))
 for s in slides:
  assert s['script'].strip() and s['sources'],s['number']
  assert not re.search(r'v(?:20|24)|R(?:79|106)',s['title']+' '+s['script']),s['number']
 counts=[{'slide':s['number'],'han_characters':len(re.findall(r'[\u3400-\u9fff]',s['script'])),'seconds':s['seconds']} for s in main]
 han=sum(x['han_characters'] for x in counts)
 assert han>=3600, f'Talk too short at 300 Han/min: {han}'
 decklist=sorted((D/'deliverables').glob('*簡報_*.pptx'),key=lambda p:p.stat().st_mtime)
 if not decklist: raise FileNotFoundError('Final presentation not built')
 deck=decklist[-1]; ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
 with zipfile.ZipFile(deck) as z:
  slidefiles=[p for p in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',p)]
  assert len(slidefiles)==23
  for s in slides:
   xml=etree.fromstring(z.read(f"ppt/notesSlides/notesSlide{s['number']}.xml"))
   t='\n'.join(xml.xpath('//a:t/text()',namespaces=ns))
   for para in s['script'].split('\n'):
    assert para in t,f"Notes incomplete: {s['number']}"
  chartcount=len([p for p in z.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml',p)])
  assert chartcount==3
 preserved=json.loads((D/'qa/protected_before.json').read_text(encoding='utf-8'))
 root=D.parents[1]
 assert all(sha(root/k)==v['sha256'] for k,v in preserved['checks'].items())
 receipt={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'narrative_sha256':sha(source),'deck':str(deck),'deck_sha256':sha(deck),'main_slides':20,'appendix_slides':3,'spoken_han_characters':han,'main_minutes_at_300_han_per_minute':han/300,'main_minutes_at_250_han_per_minute':han/250,'planned_seconds':sum(x['seconds'] for x in counts),'actual_presenter_rehearsal_completed':False,'count_excludes_titles_citations_appendices_QA':True,'complete_scripts_embedded_in_all_slide_notes':True,'native_chart_count':chartcount,'nine_protected_originals_unchanged':True,'Private_score':None,'per_slide':counts}
 (D/'qa/content_timing_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
 print(json.dumps(receipt,ensure_ascii=False))
if __name__=='__main__':main()
