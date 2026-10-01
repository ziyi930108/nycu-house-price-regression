"""Read explicit text styles from the frozen r2 PPTX packages; never edit slides."""
from pathlib import Path
from zipfile import ZipFile
from xml.etree import ElementTree as E
from datetime import datetime,timezone
import hashlib,json,posixpath
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
REF=ROOT/'outputs/v22_teaching_refinement_20261001'
NS={'p':'http://schemas.openxmlformats.org/presentationml/2006/main',
    'a':'http://schemas.openxmlformats.org/drawingml/2006/main',
    'c':'http://schemas.openxmlformats.org/drawingml/2006/chart',
    'r':'http://schemas.openxmlformats.org/officeDocument/2006/relationships'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def fill(e):
    if e is None:return None
    color=e.find('a:solidFill/a:srgbClr',NS)
    if color is None:return None
    assert all(x.tag.endswith('alpha') and x.get('val')=='100000' for x in color),'Unsupported color transform'
    return color.get('val')
def luminance(rgb):
    values=[int(rgb[i:i+2],16)/255 for i in (0,2,4)]
    v=[x/12.92 if x<=.04045 else ((x+.055)/1.055)**2.4 for x in values]
    return sum(x*w for x,w in zip(v,(.2126,.7152,.0722)))
def contrast(fg,bg):
    a,b=sorted((luminance(fg),luminance(bg)))
    return (b+.05)/(a+.05)
def record(prop,text,bg,page,kind,part):
    assert prop is not None and 'sz' in prop.attrib and bg
    fg=fill(prop);assert fg,'Unresolved foreground'
    pt=int(prop.get('sz'))/100
    return {'page':page,'kind':kind,'text':text,'part':part,'font_pt':pt,
            'bold':prop.get('b')=='1','font':prop.find('a:latin',NS).get('typeface'),
            'foreground':fg,'background':bg,'contrast':contrast(fg,bg)}
def runs(container,bg,page,kind,part):
    records=[]
    for paragraph in container.findall('a:p',NS):
        for run in paragraph.findall('a:r',NS):
            text=run.findtext('a:t',default='',namespaces=NS)
            if not text.strip():continue
            prop=run.find('a:rPr',NS)
            if prop is None:prop=paragraph.find('a:pPr/a:defRPr',NS)
            records.append(record(prop,text,bg,page,kind,part))
    return records
receipt=json.loads((REF/'delivery_receipt.json').read_text(encoding='utf-8'))
styles=[]
for item in receipt['styles']:
    p=REF/item['path'];assert sha(p)==item['sha256'];records=[];chart_count=0;table_count=0
    with ZipFile(p) as z:
        for n in range(1,24):
            part=f'ppt/slides/slide{n}.xml';slide=E.fromstring(z.read(part));bg=fill(slide.find('p:cSld/p:bg/p:bgPr',NS));assert bg
            for shape in slide.findall('p:cSld/p:spTree/p:sp',NS):
                tx=shape.find('p:txBody',NS)
                if tx is None:continue
                shape_bg=fill(shape.find('p:spPr',NS)) or bg
                rec=runs(tx,shape_bg,n,'shape',part)
                for r in rec:
                    if r['text']==f'{n:02}' and r['font_pt']==14.25:r['kind']='page_number'
                    elif n==1 and r['text']=='v22 專題報告':r['kind']='cover_version'
                records+=rec
            for tbl in slide.findall('.//a:tbl',NS):
                table_count+=1
                for cell in tbl.findall('a:tr/a:tc',NS):
                    cbg=fill(cell.find('a:tcPr',NS));assert cbg
                    records+=runs(cell.find('a:txBody',NS),cbg,n,'table',part)
            chart_refs=slide.findall('.//c:chart',NS)
            if chart_refs:
                rels=E.fromstring(z.read(f'ppt/slides/_rels/slide{n}.xml.rels'))
                rel={x.get('Id'):x.get('Target') for x in rels}
                for ref in chart_refs:
                    target=rel[ref.get('{'+NS['r']+'}id')]
                    chart_part=target.lstrip('/') if target.startswith('/') else posixpath.normpath(posixpath.join(posixpath.dirname(part),target))
                    chart=E.fromstring(z.read(chart_part));chart_count+=1
                    assert chart.find('c:spPr/a:noFill',NS) is not None
                    assert chart.find('c:chart/c:plotArea/c:spPr/a:noFill',NS) is not None
                    assert chart.find('.//c:dLblPos',NS).get('val')=='outEnd'
                    for kind,xpath in [('chart_y_axis','.//c:valAx/c:txPr'),('chart_x_axis','.//c:catAx/c:txPr'),('chart_values','.//c:dLbls/c:txPr')]:
                        tx=chart.find(xpath,NS);assert tx is not None
                        prop=tx.find('a:p/a:pPr/a:defRPr',NS)
                        records.append(record(prop,kind,bg,n,kind,chart_part))
        actual_runs=sum(len(E.fromstring(z.read(f'ppt/slides/slide{n}.xml')).findall('.//a:r',NS)) for n in range(1,24))
        assert len([x for x in records if not x['kind'].startswith('chart_')])==actual_runs
    meaningful=[x for x in records if x['kind'] not in ('page_number','cover_version')]
    minimum=min(records,key=lambda x:x['contrast'])
    styles.append({'style':item['style'],'path':p.relative_to(ROOT).as_posix(),'sha256':sha(p),'slides':23,
       'actual_native_text_runs':actual_runs,'chart_text_style_groups':chart_count*3,
       'native_tables':table_count,'native_charts':chart_count,'minimum_contrast_record':minimum,
       'minimum_meaningful_font_pt':min(x['font_pt'] for x in meaningful),
       'font_sizes_below_18pt':[x for x in records if x['font_pt']<18],
       'contrast_below_4_5':[x for x in records if x['contrast']<4.5],
       'all_records':records})
protected=json.loads((ROOT/'outputs/github_publish_v22_projection_20261001/protected_before.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/p)==h for p,h in protected.items())
result={'created_at_utc':datetime.now(timezone.utc).isoformat(),'method':'Explicit PPTX text runs and chart styles with native sRGB fills and point sizes; sRGB relative luminance ratio.',
 'formula_reference':'https://www.w3.org/WAI/WCAG22/Understanding/contrast-minimum.html',
 'scope_limit':'No WCAG conformance claim, no physical projection or user readability test; no glyph or photo-content audit. Native chart text styles are counted as groups rather than generated tick-label glyphs.',
 'styles':styles,'unresolved_text_styles':0,'protected_nine_unchanged':True,'source_pptx_unchanged':True,
 'new_slide_revision':False,'new_ML_fits':0,'Private_score':None}
with (HERE/'native_text_audit.json').open('x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps([{'style':s['style'],'min_font_pt':s['minimum_meaningful_font_pt'],
 'min_contrast':s['minimum_contrast_record']['contrast'],'runs':s['actual_native_text_runs'],
 'chart_style_groups':s['chart_text_style_groups'],'below_4_5':len(s['contrast_below_4_5'])} for s in styles]))
