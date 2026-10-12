"""Extend the preserved Word report from evidence-backed supplement.json.

Only document assets are read/written; no model fitting or price output occurs.
Run with the bundled Python runtime from any writable working directory.
"""
from pathlib import Path
from copy import deepcopy
from hashlib import sha256
from zipfile import ZipFile
import argparse, json, os, re
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

S = Path(__file__).resolve().parent
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--output-dir',type=Path,default=S.parent)
parser.add_argument('--markdown-path',type=Path)
parser.add_argument('--source-map-path',type=Path)
parser.add_argument('--qa-dir',type=Path)
args=parser.parse_args()
OUT=args.output_dir.resolve();OUT.mkdir(parents=True,exist_ok=True)
MARKDOWN=(args.markdown_path or OUT/'REPORT.md').resolve()
SOURCES=(args.source_map_path or OUT/'SOURCE_MAP.md').resolve()
QA=(args.qa_dir or OUT/'qa').resolve();QA.mkdir(parents=True,exist_ok=True)
MARKDOWN.parent.mkdir(parents=True,exist_ok=True)
REFERENCE = S / 'reference_report.docx'
EXPECTED = 'c882102537d1704b0cf5a22bcdbbb7ee84e593b77c8ba102c506e8c5bc58562e'
assert sha256(REFERENCE.read_bytes()).hexdigest() == EXPECTED
doc = Document(REFERENCE)
config = json.loads((S/'supplement.json').read_text(encoding='utf8'))
figure_assets = {}

def paragraph(text, style=None):
    p = doc.add_paragraph(text,style)
    return p

def table(headers, rows, widths):
    t = doc.add_table(rows=1, cols=len(headers))
    t.autofit = False
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for col,w in zip(t.columns,widths): col.width=Cm(w)
    for c,v in zip(t.rows[0].cells,headers): c.text=str(v)
    for row in rows:
        for c,v in zip(t.add_row().cells,row): c.text=str(v)
    for i,row in enumerate(t.rows):
        trp=row._tr.get_or_add_trPr()
        trp.append(OxmlElement('w:cantSplit'))
        if i==0: trp.append(OxmlElement('w:tblHeader'))
        for j,c in enumerate(row.cells):
            c.width=Cm(widths[j]);c.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=c._tc.get_or_add_tcPr(); borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                x=OxmlElement('w:'+edge);x.set(qn('w:val'),'single');x.set(qn('w:sz'),'4');x.set(qn('w:color'),'D9D9D9');borders.append(x)
            pr.append(borders);m=OxmlElement('w:tcMar')
            for edge,val in [('top','85'),('bottom','85'),('left','110'),('right','110')]:
                x=OxmlElement('w:'+edge);x.set(qn('w:w'),val);x.set(qn('w:type'),'dxa');m.append(x)
            pr.append(m);shade=OxmlElement('w:shd');shade.set(qn('w:fill'),'DCE6F1' if i==0 else ('F5F7FA' if i%2==0 else 'FFFFFF'));pr.append(shade)
            for p in c.paragraphs:
                p.paragraph_format.line_spacing=Pt(15.5);p.paragraph_format.space_after=Pt(1)
                p.alignment=WD_ALIGN_PARAGRAPH.CENTER if i==0 or len(c.text)<19 else WD_ALIGN_PARAGRAPH.LEFT
                for r in p.runs: r.font.size=Pt(10.5);r.bold=i==0
    paragraph('').paragraph_format.space_after=Pt(0)
    return t

def block(data):
    kind=data['kind']
    if kind=='p':
        p=paragraph(data['text']);p.paragraph_format.keep_with_next=data.get('keep',False)
    elif kind=='h2': paragraph(data['text'],'Heading 2')
    elif kind=='table': table(data['headers'],data['rows'],data['widths'])
    elif kind=='image':
        path=S/data['path']; p=paragraph('');p.alignment=WD_ALIGN_PARAGRAPH.CENTER
        p.paragraph_format.line_spacing=1.0;p.paragraph_format.keep_with_next=True
        shape=p.add_run().add_picture(str(path),width=Cm(data.get('width_cm',16.8)))
        shape._inline.docPr.set('descr',data['alt'])
        rid=shape._inline.graphic.graphicData.pic.blipFill.blip.embed
        figure_assets[rid]=data['path']
        c=paragraph(data['caption'],'Caption');c.alignment=WD_ALIGN_PARAGRAPH.CENTER
    elif kind=='code':
        for line in data['text'].splitlines():
            p=paragraph(line);p.paragraph_format.line_spacing=Pt(14.5);p.paragraph_format.space_after=Pt(0)
            p.paragraph_format.keep_with_next=True
            for run in p.runs:run.font.name='Consolas';run.font.size=Pt(9)
        paragraph('')
    else: raise ValueError(kind)

# Each insertion uses an existing chapter's preceding page-break as its anchor.
# New blocks are built with source styles, then moved into the original body.
for group in config['insertions']:
    anchor=next(p._p for p in doc.paragraphs if p.text==group['before'])
    previous=anchor.getprevious()
    assert previous is not None and previous.find('.//'+qn('w:br')) is not None
    for page in group['pages']:
        start=list(doc.element.body)
        paragraph(page['title'],'Heading 1')
        for b in page['blocks']:block(b)
        doc.add_page_break()
        old_ids={id(n) for n in start}
        added=[n for n in list(doc.element.body) if id(n) not in old_ids and n.tag!=qn('w:sectPr')]
        for node in added:anchor.addprevious(node)

# Replace the two previous uses of the word the user asked to remove.
for p in doc.paragraphs:
    for run in p.runs:
        old=run.text
        new=old.replace('方法研究','方法比較').replace('研究','開發')
        new=new.replace('六模型融合結果需看前頁的整體驗證','六模型融合結果需看完整融合的整體驗證')
        new=new.replace('圖1 單一參考單價神經網路的診斷曲線','圖5 單一參考單價神經網路的診斷曲線')
        new=new.replace('此數值來自使用者文字回報，未另登入平台核對實際上傳檔案的內容；Private仍未知。','此數值原由使用者回報，已由本次截圖及登入Kaggle頁面核對；尚未核對平台上傳檔案bytes，Private未知。')
        new=new.replace('[5] 最終選定與使用者公開分數回報。','[5] 最終選定、使用者公開分數回報與本次平台頁面核對。')
        new=new.replace('閱讀順序：資料清理與轉換 → 特徵 → 模型與融合 → 訓練 → 時間驗證 → 曲線與結果 → 最終答案。','閱讀順序：程式流程 → 資料清理與分布 → 特徵 → 模型與融合 → 訓練 → 時間驗證與誤差分析 → 最終答案。')
        if new!=old:run.text=new

number=0
for p in doc.paragraphs:
    if p.style.name=='Heading 1' and p.text!='摘要':
        number+=1
        old=re.sub(r'^\d+\s+','',p.text)
        p.text=f'{number} {old}'

source_anchor=next(p._p for p in doc.paragraphs if p.text.startswith('程式與證據入口：'))
for text in ['[7] 清理後train的城市總價描述統計與樣本分布。',
             '[8] 封存完整融合驗證預測的列序、誤差、分群及匿名案例核對。']:
    p=paragraph(text);p.paragraph_format.space_after=Pt(2)
    for r in p.runs:r.font.size=Pt(9.5)
    source_anchor.addprevious(p._p)

# Preserve the existing graphic relationship in the markdown export.
for p in doc.paragraphs:
    for blip in p._p.findall('.//'+qn('a:blip')):
        rid=blip.get(qn('r:embed'))
        if rid not in figure_assets:figure_assets[rid]='assets/training_curve.png'

dest=OUT/'台灣房屋銷售價格預測挑戰賽_正式報告.docx'
doc.core_properties.title='台灣房屋銷售價格預測挑戰賽'
doc.core_properties.author='廖子頤'
doc.save(dest)

def cell_text(c):return c.text.replace('\n','<br>').replace('|','\\|')
md=[];in_code=False
for node in doc.element.body:
    if node.tag==qn('w:p'):
        p=Paragraph(node,doc)
        code=bool(p.text) and bool(p.runs) and all(r.font.name=='Consolas' and r.font.size and r.font.size.pt==9 for r in p.runs if r.text)
        if code:
            if not in_code:md+=['```python'];in_code=True
            md += [p.text]
            continue
        if in_code:md+=['```',''];in_code=False
        drawings=node.findall('.//'+qn('a:blip'))
        if drawings:
            for b in drawings:
                rid=b.get(qn('r:embed')); asset=figure_assets[rid]
                link=Path(os.path.relpath(S/asset,MARKDOWN.parent)).as_posix()
                md += [f'![分析圖]({link})','']
        elif p.text:
            prefix={'Title':'# ','Heading 1':'## ','Heading 2':'### '}.get(p.style.name,'')
            md += [prefix+p.text,'']
    elif node.tag==qn('w:tbl'):
        if in_code:md+=['```',''];in_code=False
        t=Table(node,doc);rows=[[cell_text(c) for c in r.cells] for r in t.rows]
        md += ['| '+' | '.join(rows[0])+' |','| '+' | '.join(['---']*len(rows[0]))+' |']
        md += ['| '+' | '.join(row)+' |' for row in rows[1:]]+['']
source_link=Path(os.path.relpath(SOURCES,MARKDOWN.parent)).as_posix()
md += [f'詳細證據與數據來源見 [來源對照]({source_link})。','']
MARKDOWN.write_text('\n'.join(md),encoding='utf8')
text='\n'.join(p.text for p in doc.paragraphs)+'\n'+'\n'.join(c.text for t in doc.tables for r in t.rows for c in r.cells)
assert not re.search(r'v25|本研究|研究|報告日期|方法與實驗報告',text,re.I)
assert '公開MAE約199.58萬元' not in '\n'.join(p.text for p in doc.paragraphs[:9])
assert sha256(REFERENCE.read_bytes()).hexdigest()==EXPECTED

with ZipFile(REFERENCE) as old, ZipFile(dest) as new:
    original={n:sha256(old.read(n)).hexdigest() for n in old.namelist()}
    current={n:sha256(new.read(n)).hexdigest() for n in new.namelist()}
    changed=[n for n in original if original[n]!=current.get(n)]
    preserved=['word/styles.xml','word/stylesWithEffects.xml','word/numbering.xml','word/theme/theme1.xml','word/footer1.xml','word/media/image1.png']
    assert all(original.get(n)==current.get(n) for n in preserved if n in original),changed
manifest={'docx':dest.name,'sha256':sha256(dest.read_bytes()).hexdigest(),'reference_sha256':EXPECTED,
          'chapters':number,'tables':len(doc.tables),'images':len(doc.inline_shapes),'changed_original_parts':changed,
          'new_parts':[n for n in current if n not in original],'preserve_only_parts':preserved,
          'model_training':False,'submission_csv_changed':False,'Kaggle_operation':False,
          'rendered_and_visually_reviewed':False}
(QA/'build.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding='utf8')
(QA/'report_text.txt').write_text(text,encoding='utf8')
print(json.dumps(manifest,ensure_ascii=False,indent=2))
