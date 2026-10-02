"""Build an A4 Traditional Chinese report PDF from the project's REPORT.md.

Uses ReportLab and installed Microsoft JhengHei fonts. It does not run models.
The Markdown report remains the editable and exact-method source.
"""
from __future__ import annotations
import argparse
import hashlib
import html
import json
from pathlib import Path
import re

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas
from reportlab.platypus import (
    BaseDocTemplate, Frame, KeepTogether, PageTemplate, Paragraph,
    Spacer, Table, TableStyle,
)

FONT_ROOT = Path('C:/Windows/Fonts')
PAGE_W, PAGE_H = A4
MARGIN = 18 * mm
CONTENT_W = PAGE_W - 2 * MARGIN
INK = colors.HexColor('#233E37')
ACCENT = colors.HexColor('#31644F')
MUTED = colors.HexColor('#586761')
PALE = colors.HexColor('#EEF3EF')
LINE = colors.HexColor('#CCD6CE')


def register_fonts(font_root: Path) -> None:
    pdfmetrics.registerFont(TTFont('JhengHei', str(font_root / 'msjh.ttc'), subfontIndex=0))
    pdfmetrics.registerFont(TTFont('JhengHei-Bold', str(font_root / 'msjhbd.ttc'), subfontIndex=0))
    pdfmetrics.registerFontFamily('JhengHei', normal='JhengHei', bold='JhengHei-Bold', italic='JhengHei', boldItalic='JhengHei-Bold')


def styles():
    base = dict(fontName='JhengHei', fontSize=10.1, leading=16.2,
                textColor=INK, wordWrap='CJK', splitLongWords=True,
                allowWidows=0, allowOrphans=0)
    out = {'body': ParagraphStyle('body', spaceAfter=7.0, **base)}
    out['title'] = ParagraphStyle('title', fontName='JhengHei-Bold', fontSize=21,
                                  leading=28, textColor=INK, spaceAfter=8, keepWithNext=True,
                                  wordWrap='CJK')
    out['subtitle'] = ParagraphStyle('subtitle', fontName='JhengHei', fontSize=9.2,
                                    leading=14, textColor=MUTED, spaceAfter=17, keepWithNext=True)
    out['h2'] = ParagraphStyle('h2', fontName='JhengHei-Bold', fontSize=14,
                              leading=20, textColor=ACCENT, spaceBefore=13,
                              spaceAfter=7, keepWithNext=True, wordWrap='CJK')
    out['h3'] = ParagraphStyle('h3', fontName='JhengHei-Bold', fontSize=11.2,
                              leading=17, textColor=INK, spaceBefore=9,
                              spaceAfter=5, keepWithNext=True, wordWrap='CJK')
    out['cell'] = ParagraphStyle('cell', fontName='JhengHei', fontSize=8.7,
                                leading=13.4, textColor=INK, wordWrap='CJK',
                                splitLongWords=True, allowWidows=1, allowOrphans=1)
    out['cellhead'] = ParagraphStyle('cellhead', parent=out['cell'], fontName='JhengHei-Bold', textColor=colors.white)
    out['code'] = ParagraphStyle('code', fontName='JhengHei', fontSize=8.9,
                                leading=13.6, textColor=INK, wordWrap='CJK',
                                splitLongWords=True)
    out['equation'] = ParagraphStyle('equation', parent=out['body'], fontSize=10.3,
                                    leading=17, alignment=TA_CENTER, spaceAfter=0)
    out['list'] = ParagraphStyle('list', parent=out['body'], leftIndent=12, firstLineIndent=-10,
                                spaceAfter=5)
    return out


def normalize(s: str) -> str:
    return s.replace('\ufeff', '').replace('\u2011', '-').replace('\u2013', '-').replace('\u2014', '-').replace('\u2212', '-')


def inline_math(s: str) -> str:
    s = s.replace(r'\hat y_i', 'y_pred,i').replace(r'\hat y', 'y_pred')
    s = s.replace(r'\Delta t_i', 'Delta t_i')
    return s


def inline(s: str) -> str:
    # Only formatting from this local Markdown is interpreted. No external fetches.
    s = normalize(s)
    s = re.sub(r'\$([^$]+)\$', lambda m: inline_math(m.group(1)), s)
    s = html.escape(s)
    s = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', s)
    s = re.sub(r'`([^`]+)`', r'<font color="#315947">\1</font>', s)
    s = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\1', s)
    return s


class NumberedCanvas(Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._page_states = []
    def showPage(self):
        self._page_states.append(dict(self.__dict__))
        self._startPage()
    def save(self):
        total = len(self._page_states)
        for state in self._page_states:
            self.__dict__.update(state)
            self.saveState()
            self.setStrokeColor(LINE)
            self.setLineWidth(.45)
            self.line(MARGIN, 14.4 * mm, PAGE_W - MARGIN, 14.4 * mm)
            self.setFont('JhengHei', 7.3)
            self.setFillColor(MUTED)
            self.drawString(MARGIN, 10.2 * mm, '臺灣房屋總價預測 | 技術報告 | 提交模型 v22')
            self.drawRightString(PAGE_W - MARGIN, 10.2 * mm, f'{self._pageNumber} / {total}')
            self.restoreState()
            super().showPage()
        super().save()


class ReportDoc(BaseDocTemplate):
    def __init__(self, filename: str):
        super().__init__(filename, pagesize=A4, leftMargin=MARGIN, rightMargin=MARGIN,
                         topMargin=20*mm, bottomMargin=21*mm,
                         title='臺灣房屋總價預測：近期成交加權與多模型回歸',
                         author='房屋總價預測研究專案')
        frame = Frame(MARGIN, 21*mm, CONTENT_W, PAGE_H-41*mm,
                      leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates(PageTemplate(id='Report', frames=[frame], onPage=self.header))
        self.section_pages = []
    def header(self, canvas, doc):
        canvas.saveState()
        canvas.setFont('JhengHei', 7.5)
        canvas.setFillColor(MUTED)
        canvas.drawString(MARGIN, PAGE_H-12.0*mm, 'TAIWAN HOUSE PRICE REGRESSION')
        canvas.drawRightString(PAGE_W-MARGIN, PAGE_H-12.0*mm, '已選定提交模型 | Private 尚未知')
        canvas.setLineWidth(.45)
        canvas.setStrokeColor(LINE)
        canvas.line(MARGIN, PAGE_H-14.3*mm, PAGE_W-MARGIN, PAGE_H-14.3*mm)
        canvas.restoreState()
    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and flowable.style.name in ('h2', 'h3'):
            self.section_pages.append({'heading': flowable.getPlainText(), 'page': self.page})


def source_table(rows):
    # PDF citations stay legible. The Markdown companion holds the full path manifest.
    short = {
        'E1': 'v22_method_and_code.md',
        'E2': 'round79_report.md\nround79/results.json\nround79/paired_vs_v20/results.json',
        'E3': 'round83_report.md\nround83_delivery.md\nround83_final.py',
        'E4': 'round84_report.md\nround84/decision.json',
        'E5': 'public_submissions_v19_v24_20260927.json\n同名 Kaggle 成績截圖',
        'E6': 'prepare_data.py / model_common.py\nhouse_features.py / prepare_features.py',
        'E7': 'round19_unit_geo.py\nround79_recency_unit.py / round79_final_unit.py\nround79/final_model/model/metadata.json',
        'E8': 'round47 / round48 NN trainer\nround54_numeric_encode.py\nround60_recency_price_prior.py\nround63_recency_weighted_prior_train.py',
        'E9': 'round12 / round46 log 模型\nround71_final.py / round46_trial_release.py\nround2_neighborhood.py\nround8_spatial_reference.py / round79_final.py',
        'E10': 'round79/final/trial_release.json\nround79/final_export/release.json\nround79_release.py',
        'E11': 'docs/REPRODUCIBILITY.md\nevidence/reproduction/summary.json\nevidence/reproduction/integration_verified.json\n歷史：github_reproducibility_audit_r83.md',
        'E12': 'geography/ATTRIBUTION.md\ngeography/*.download.json\nREADME_round6.md',
    }
    return [rows[0]] + [[row[0], row[1], short.get(row[0], row[2])] for row in rows[1:]]


def make_table(rows, st):
    n = len(rows[0])
    source_index = rows[0][0] == '編號' and any(r[0] == 'E1' for r in rows[1:])
    vertical_padding = 4 if source_index else 6
    if rows[0][0] == '編號' and any(r[0] == 'E1' for r in rows[1:]):
        rows = source_table(rows)
        fractions = [.07, .31, .62]
    elif rows[0][0] == 'query 期間' and n == 5:
        fractions = [.30, .175, .175, .16, .19]
    elif rows[0][0] == 'query 期間' and n == 3:
        fractions = [.46, .30, .24]
    elif rows[0][0] == '期別' and n == 4:
        fractions = [.17, .43, .20, .20]
    elif rows[0][0] == '年度' and n == 3:
        fractions = [.16, .42, .42]
    elif rows[0][0] == '檔案':
        fractions = [.44, .13, .43]
    elif rows[0][0] in ('比較', '模式') and n == 3:
        fractions = [.16, .37, .47]
    elif rows[0][0] == '功能':
        fractions = [.25, .75]
    elif n == 2:
        fractions = [.30, .70]
    elif n == 3:
        fractions = [.46, .27, .27]
    else:
        fractions = [1/n] * n
    cells = []
    for r, row in enumerate(rows):
        cells.append([Paragraph(inline(c).replace('v22 相對 v20 降低', 'v22 改善<br/>相對 v20').replace('\n', '<br/>'), st['cellhead'] if r == 0 else st['cell']) for c in row])
    result = Table(cells, colWidths=[CONTENT_W*f for f in fractions], repeatRows=1,
                   hAlign='LEFT', splitByRow=1, spaceBefore=4, spaceAfter=9)
    cmds = [('BACKGROUND',(0,0),(-1,0),ACCENT), ('VALIGN',(0,0),(-1,-1),'TOP'),
            ('LEFTPADDING',(0,0),(-1,-1),7), ('RIGHTPADDING',(0,0),(-1,-1),7),
            ('TOPPADDING',(0,0),(-1,-1),vertical_padding), ('BOTTOMPADDING',(0,0),(-1,-1),vertical_padding),
            ('LINEBELOW',(0,0),(-1,0),.6,ACCENT),
            ('LINEBELOW',(0,1),(-1,-1),.3,LINE)]
    for row in range(1,len(rows)):
        if row%2:
            cmds.append(('BACKGROUND',(0,row),(-1,row),PALE))
    result.setStyle(TableStyle(cmds))
    return KeepTogether([result]) if len(rows) <= 10 else result


def box(paras, width=CONTENT_W):
    t = Table([[paras]], colWidths=[width], spaceBefore=4, spaceAfter=10)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,-1),PALE),
        ('BOX',(0,0),(-1,-1),.4,LINE), ('LEFTPADDING',(0,0),(-1,-1),10),
        ('RIGHTPADDING',(0,0),(-1,-1),10), ('TOPPADDING',(0,0),(-1,-1),8),
        ('BOTTOMPADDING',(0,0),(-1,-1),8)]))
    return t


def equation(raw, st):
    if r'\operatorname{MAE}' in raw:
        lines = ['MAE = (1 / n) × Σ |y<sub>i</sub> - y_pred,<sub>i</sub>|']
    elif 'date,raw' in raw:
        lines = ['w<sub>date, raw, i</sub> = 2<super>-Δt<sub>i</sub> / 365</super>']
    elif r'\frac{A_i}{100}' in raw:
        lines = ['(A<sub>i</sub> / 100) × w<sub>date, i</sub> × |y<sub>i</sub> / (10,000A<sub>i</sub>) - z<sub>i</sub>|',
                 '= (w<sub>date, i</sub> / 1,000,000) × |y<sub>i</sub> - 10,000A<sub>i</sub> z<sub>i</sub>|']
    else:
        lines = [html.escape(raw)]
    return box([Paragraph(v, st['equation']) for v in lines])


def parse_markdown(source: str, st):
    lines = source.splitlines()
    story=[]
    i=0
    while i < len(lines):
        line=lines[i].strip()
        if not line:
            i += 1
            continue
        if line.startswith('```'):
            block=[]; i+=1
            while i<len(lines) and not lines[i].strip().startswith('```'):
                block.append(lines[i]); i+=1
            story.append(box([Paragraph(html.escape(normalize(x)).replace(' ', '&#160;'), st['code']) for x in block if x.strip()]))
            i+=1
            continue
        if line == '$$':
            block=[];i+=1
            while i<len(lines) and lines[i].strip() != '$$':
                block.append(lines[i]); i+=1
            story.append(equation('\n'.join(block),st)); i+=1
            continue
        if line.startswith('|'):
            rows=[]
            while i<len(lines) and lines[i].strip().startswith('|'):
                row=[v.strip() for v in lines[i].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?',v or 'x') for v in row): rows.append(row)
                i+=1
            story.append(make_table(rows,st));continue
        if line.startswith('# '):
            story.append(Paragraph(inline(line[2:]),st['title']))
            story.append(Paragraph('技術報告 | 已選定提交模型 v22 | 以新臺幣總價 MAE 評估',st['subtitle']))
            i+=1;continue
        if line.startswith('## '):
            story.append(Paragraph(inline(line[3:]),st['h2'])); i+=1;continue
        if line.startswith('### '):
            story.append(Paragraph(inline(line[4:]),st['h3'])); i+=1;continue
        if line.startswith('- ') or re.match(r'^\d+\. ',line):
            prefix='• ' if line.startswith('- ') else ''
            value=line[2:] if line.startswith('- ') else line
            story.append(Paragraph(prefix+inline(value),st['list'])); i+=1;continue
        text=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|\||```|\$\$|- |\d+\. )',lines[i].strip()):
            text.append(lines[i].strip());i+=1
        story.append(Paragraph(inline(' '.join(text)),st['body']))
    combined=[]
    idx=0
    while idx<len(story):
        current=story[idx]
        if (isinstance(current,Paragraph) and current.style.name in ('h2','h3')
            and idx+1<len(story) and isinstance(story[idx+1],KeepTogether)):
            combined.append(KeepTogether([current]+story[idx+1]._content))
            idx+=2
        else:
            combined.append(current);idx+=1
    return combined


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--font-dir',type=Path,default=FONT_ROOT)
    args=parser.parse_args()
    if args.output.exists() or args.output.with_suffix('.build.json').exists():
        raise FileExistsError('Output already exists; select a new output path')
    register_fonts(args.font_dir)
    text=args.input.read_text(encoding='utf-8-sig')
    if not text.strip(): raise ValueError('Empty report')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    doc=ReportDoc(str(args.output))
    story=parse_markdown(text,styles())
    doc.build(story,canvasmaker=NumberedCanvas)
    from pypdf import PdfReader
    reader=PdfReader(args.output)
    extracted='\n'.join(p.extract_text() or '' for p in reader.pages)
    required=['467,221','1,998,568.5706','Private','v24','138.716']
    missing=[s for s in required if s not in extracted]
    if missing: raise RuntimeError(f'Missing PDF text: {missing}')
    receipt={'input':str(args.input),'output':str(args.output),'input_sha256':hashlib.sha256(args.input.read_bytes()).hexdigest(),
             'output_sha256':hashlib.sha256(args.output.read_bytes()).hexdigest(),'pages':len(reader.pages),
             'font':'Microsoft JhengHei','page_size':'A4','section_pages':doc.section_pages,
             'text_checks':{s:s in extracted for s in required},'visual_qa':'pending_png_inspection'}
    args.output.with_suffix('.build.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'pages':len(reader.pages),'output':str(args.output),'text_checks_passed':not missing},ensure_ascii=False))

if __name__=='__main__':main()


