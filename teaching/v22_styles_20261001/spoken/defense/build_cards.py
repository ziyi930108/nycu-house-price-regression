"""Build four printable defense cue pages from the approved answers; no training."""
from pathlib import Path
from datetime import datetime,timezone
from xml.sax.saxutils import escape
import argparse,hashlib,json,re,subprocess,time
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import Paragraph
from pypdf import PdfReader

HERE=Path(__file__).resolve().parent
TITLE='臺灣房價預測 v22 答辯提示卡'
NAME='臺灣房價預測_v22_答辯提示卡'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
 with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
def clean(s):return re.sub(r'\s+','',s)
def main():
 ap=argparse.ArgumentParser(description=__doc__)
 ap.add_argument('--content',type=Path,default=HERE/'questions_approved.json');ap.add_argument('--output',type=Path,required=True)
 ap.add_argument('--font',type=Path,required=True);ap.add_argument('--bold-font',type=Path,required=True);ap.add_argument('--pdftoppm',type=Path,required=True)
 a=ap.parse_args();data=json.loads(a.content.read_text(encoding='utf-8'));questions=data['questions']
 assert [q['number'] for q in questions]==list(range(1,17))
 assert not a.output.exists();out=a.output.resolve()
 for src in (a.content.resolve(),a.font.resolve(),a.bold_font.resolve(),a.pdftoppm.resolve()):
  assert src.is_file() and not src.is_relative_to(out)
 out.mkdir(parents=True);(out/'renders').mkdir();started=time.perf_counter()
 pdfmetrics.registerFont(TTFont('JH',str(a.font),subfontIndex=0));pdfmetrics.registerFont(TTFont('JHB',str(a.bold_font),subfontIndex=0));pdfmetrics.registerFontFamily('JH',normal='JH',bold='JHB')
 dark=colors.HexColor('#16354A');muted=colors.HexColor('#435768');blue=colors.HexColor('#16538C')
 qstyle=ParagraphStyle('question',fontName='JHB',fontSize=16,leading=21,wordWrap='CJK',textColor=dark)
 sstyle=ParagraphStyle('say',fontName='JH',fontSize=13.2,leading=18.5,wordWrap='CJK',textColor=colors.HexColor('#18232D'))
 fstyle=ParagraphStyle('follow',fontName='JH',fontSize=11.7,leading=16.5,wordWrap='CJK',textColor=muted)
 width,height=A4;margin=42;usable=width-2*margin;pdf=out/(NAME+'.pdf');md=out/(NAME+'.md')
 c=canvas.Canvas(str(pdf),pagesize=A4,pageCompression=1,invariant=1);c.setTitle(TITLE);c.setAuthor('臺灣房屋總價預測專題');c.setSubject('16題口語短答與追問補充，問答不计入主講時長')
 groups=['任務、分數與驗證','資料處理與權重','先驗、資料來源與過擬合','組合、Private與重現']
 layout=[];markdown=['# '+TITLE,'','2026-10-01。先回答第一段，老師追問再補第二段。問答不計入至少12分鐘主講。','']
 for page in range(4):
  c.setFillColor(dark);c.setFont('JHB',23);c.drawString(margin,height-45,'v22 答辯提示卡')
  c.setFont('JH',11.5);c.drawString(margin,height-66,groups[page]);c.setFillColor(muted);c.setFont('JH',9.5);c.drawRightString(width-margin,height-66,f'第{page+1}頁 / 共4頁')
  c.drawString(margin,height-85,'先回答第一段，追問再補充。問答不計入主講時間。')
  top=height-111;slot=157
  for q in questions[page*4:page*4+4]:
   c.bookmarkPage('q'+str(q['number']));c.addOutlineEntry(f"{q['number']:02} {q['question']}",'q'+str(q['number']),level=0,closed=False)
   y=top
   for kind,text,style,gap in [('question',f"{q['number']:02}  {q['question']}",qstyle,7),('say','<b>先這樣答：</b>'+escape(q['say']),sstyle,5),('follow','<b>追問再補：</b>'+escape(q['follow_up']),fstyle,0)]:
    paragraph=Paragraph(escape(text) if kind=='question' else text,style);w,h=paragraph.wrap(usable,slot)
    y-=h;paragraph.drawOn(c,margin,y);layout.append({'question':q['number'],'page':page+1,'part':kind,'top':y+h,'bottom':y,'height':h,'font_size':style.fontSize})
    y-=gap
   if y<top-slot+14:raise ValueError(f"Question {q['number']} exceeds allotted height; shorten text or use a new layout revision")
   c.setFont('JH',9);c.setFillColor(blue);c.drawRightString(width-margin,top-slot+7,'對照簡報：'+q['slide_hint'])
   if q['number']%4:c.setStrokeColor(colors.HexColor('#D8E0E7'));c.setLineWidth(.5);c.line(margin,top-slot,width-margin,top-slot)
   top-=slot
   markdown.extend([f"## {q['number']:02} {q['question']}",'','**先這樣答：** '+q['say'],'','**追問再補：** '+q['follow_up'],'','對照簡報：'+q['slide_hint'],''])
  c.setFillColor(muted);c.setFont('JH',8.5);c.drawString(margin,48,'依據：v22完整口語講稿與排練問答（2026-10-01）。')
  c.drawString(margin,33,'Public：1,998,568.5706元。Private：未知。保存模型推論已驗證，從零重訓尚未驗證。')
  c.showPage()
 c.save();md.write_text('\n'.join(markdown)+'\n',encoding='utf-8')
 reader=PdfReader(pdf);assert len(reader.pages)==4 and len(reader.outline)==16
 texts=[p.extract_text() or '' for p in reader.pages]
 for q in questions:
  actual=clean(texts[(q['number']-1)//4])
  for key in ('question','say','follow_up','slide_hint'):assert clean(q[key]) in actual,(q['number'],key)
 command=[str(a.pdftoppm),'-r','120','-png',str(pdf),str(out/'renders/page')]
 p=subprocess.run(command,capture_output=True);(out/'render.stdout.log').write_bytes(p.stdout);(out/'render.stderr.log').write_bytes(p.stderr);assert p.returncode==0
 receipt={'status':'built_pending_visual_review','created_at_utc':datetime.now(timezone.utc).isoformat(),'pdf':pdf.name,'pdf_sha256':sha(pdf),'markdown':md.name,'markdown_sha256':sha(md),'pages':4,'bookmarks':16,'all_16_questions_and_both_answers_extracted':True,'page_size_points':list(A4),'minimum_answer_font_pt':11.7,'layout':layout,'builder_sha256':sha(Path(__file__)),'content_sha256':sha(a.content),'fonts':{'regular':str(a.font),'bold':str(a.bold_font)},'font_sha256':{'regular':sha(a.font),'bold':sha(a.bold_font)},'elapsed_seconds':time.perf_counter()-started,'new_ML_fits':0}
 save(out/'build_receipt.json',receipt);print(json.dumps({k:v for k,v in receipt.items() if k not in ('layout','fonts','font_sha256')},ensure_ascii=False))

if __name__=='__main__':main()
