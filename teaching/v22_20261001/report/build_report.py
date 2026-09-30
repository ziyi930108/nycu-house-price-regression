"""Build beginner v22 report from structured content. No model training.
Usage: bundled-python build_report.py; reportlab and pypdf are required.
"""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
from xml.sax.saxutils import escape
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from pypdf import PdfReader
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
TITLE='臺灣房價預測 v22 零基礎報告'
NAME='臺灣房價預測_v22_零基礎報告'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output-dir',type=Path,default=ROOT/'deliverables')
    ap.add_argument('--font',type=Path,default=Path('C:/Windows/Fonts/msjh.ttc'))
    ap.add_argument('--bold-font',type=Path,default=Path('C:/Windows/Fonts/msjhbd.ttc'))
    a=ap.parse_args()
    content=json.loads((HERE/'report_content.json').read_text(encoding='utf-8-sig'))
    pdfmetrics.registerFont(TTFont('JH',str(a.font),subfontIndex=0))
    pdfmetrics.registerFont(TTFont('JHB',str(a.bold_font),subfontIndex=0))
    pdfmetrics.registerFontFamily('JH',normal='JH',bold='JHB')
    body=ParagraphStyle('body',fontName='JH',fontSize=10.4,leading=16.4,spaceAfter=8,wordWrap='CJK',textColor=colors.HexColor('#222222'))
    title=ParagraphStyle('title',parent=body,fontName='JHB',fontSize=22,leading=29,spaceAfter=17,textColor=colors.black)
    h2=ParagraphStyle('h2',parent=body,fontName='JHB',fontSize=13.2,leading=19,spaceBefore=8,spaceAfter=6,textColor=colors.black)
    small=ParagraphStyle('small',parent=body,fontSize=8.5,leading=12.5,spaceAfter=5,textColor=colors.HexColor('#555555'))
    cell=ParagraphStyle('cell',parent=body,fontSize=9.3,leading=14,spaceAfter=0)
    head=ParagraphStyle('head',parent=cell,fontName='JHB',textColor=colors.white)
    code=ParagraphStyle('code',parent=body,fontName='JH',fontSize=8.5,leading=12.8,leftIndent=8,rightIndent=8,spaceBefore=4,spaceAfter=9,backColor=colors.HexColor('#F2F4F5'),borderPadding=8,wordWrap='CJK')
    def para(s,st=body): return Paragraph(escape(s).replace('\n','<br/>'),st)
    story=[];md=['# '+TITLE,'','2026-10-01｜課程專題正式說明與自學版本','','本報告說明選定提交模型 v22 的完整流程、設計理由、實際成果與證據界線。','']
    for i,page in enumerate(content['pages']):
        if i:story.append(PageBreak())
        story.append(para(page['title'] if i==0 else f'{i+1:02d}  '+page['title'],title))
        if i:md.extend(['## '+page['title'],''])
        for block in page['blocks']:
            kind=block['type']
            if kind=='p':story.append(para(block['text']));md.extend([block['text'],''])
            elif kind=='h':story.append(para(block['text'],h2));md.extend(['### '+block['text'],''])
            elif kind=='code':story.append(para(block['text'],code));md.extend(['```'+block.get('lang','text'),block['text'],'```',''])
            elif kind=='table':
                table_cell=ParagraphStyle('sourcecell',parent=cell,fontSize=8.2,leading=11.4) if i==13 else cell
                table_head=ParagraphStyle('sourcehead',parent=table_cell,fontName='JHB',textColor=colors.white) if i==13 else head
                rows=[[para(str(v),table_head if ri==0 else table_cell) for v in row] for ri,row in enumerate(block['rows'])]
                widths=[(A4[0]-100)*x for x in block['widths']]
                t=Table(rows,colWidths=widths,hAlign='LEFT',repeatRows=1)
                t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#173C50')),('GRID',(0,0),(-1,-1),.45,colors.HexColor('#D9D9D9')),('VALIGN',(0,0),(-1,-1),'MIDDLE'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),5 if i==13 else 7),('BOTTOMPADDING',(0,0),(-1,-1),5 if i==13 else 7),('ROWBACKGROUNDS',(0,1),(-1,-1),[colors.white,colors.HexColor('#F5F7F8')])]))
                story.extend([t,Spacer(1,10)]);md.append('| '+' | '.join(block['rows'][0])+' |');md.append('| '+' | '.join(['---']*len(block['rows'][0]))+' |');md.extend('| '+' | '.join(str(v).replace('\n','<br>') for v in row)+' |' for row in block['rows'][1:]);md.append('')
            else:raise ValueError(kind)
        source='依據：'+page['sources']+'。完整路徑見末頁。'
        story.append(Spacer(1,4));story.append(para(source,small));md.extend([source,''])
    pdf=a.output_dir/(NAME+'.pdf');markdown=a.output_dir/(NAME+'.md');a.output_dir.mkdir(parents=True,exist_ok=True)
    def page_chrome(c,doc):
        c.setFont('JH',8);c.setFillColor(colors.HexColor('#666666'));c.drawString(50,A4[1]-31,'臺灣房屋總價預測  |  v22 學習與報告版');c.drawString(50,29,'2026-10-01  |  Public 為截圖回報，Private 尚未知');c.drawRightString(A4[0]-50,29,str(doc.page))
    doc=SimpleDocTemplate(str(pdf),pagesize=A4,leftMargin=50,rightMargin=50,topMargin=55,bottomMargin=53,title=TITLE,author='臺灣房屋價格預測專題',subject='完整流程、設計理由與可重現範圍')
    doc.build(story,onFirstPage=page_chrome,onLaterPages=page_chrome);markdown.write_text('\n'.join(md),encoding='utf-8')
    reader=PdfReader(pdf);texts=[p.extract_text() or '' for p in reader.pages]
    receipt={'status':'built_pending_visual_review','created_utc':datetime.now(timezone.utc).isoformat(),'pdf':str(pdf),'markdown':str(markdown),'pdf_sha256':digest(pdf),'markdown_sha256':digest(markdown),'pages':len(reader.pages),'planned_pages':len(content['pages']),'page_text_lengths':[len(t) for t in texts],'builder_sha256':digest(__file__),'content_sha256':digest(HERE/'report_content.json'),'fonts':[str(a.font),str(a.bold_font)],'render_required':True,'new_model_fits':0,'notes':'Text extraction and page count do not constitute visual QA.'}
    (HERE/'build_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8');(HERE/'extracted_text.txt').write_text('\n\n'.join(texts),encoding='utf-8')
    assert len(reader.pages)==len(content['pages']),receipt
    assert all(len(t)>250 for t in texts),receipt
    for phrase in ['1,998,568.5706','467,221','47,073','365','Private','OOF','median']:assert phrase in ''.join(texts),phrase
    print(json.dumps(receipt,ensure_ascii=False,indent=2))
if __name__=='__main__':main()
