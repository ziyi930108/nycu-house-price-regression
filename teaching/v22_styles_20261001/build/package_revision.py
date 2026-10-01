"""Check the two-page teaching revision against all three preserved originals."""
from pathlib import Path
from datetime import datetime,timezone
from xml.etree import ElementTree as ET
import hashlib,json,re,zipfile
from PIL import Image

D=Path(__file__).resolve().parents[1]
ROOT=D.parents[1]
styles=[('A','白底學術','v22_full_style_20261001','c4e7359ac0e222713df00337ff350f8bf7d1e1d26a4798712240bbb70e866200'),
        ('B','深色科技','v22_dark_style_20261001','c3ea6fc9de9c838f13866edbd0077cc542ec04e6b93bb6c90a95ca4cf939e387'),
        ('C','暖色雜誌','v22_warm_style_20261001','d893b695272babf0cc13700a701df0627deab8372609de79d414e2ae2e9e76fc')]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
narrative=json.loads((D/'content/narrative.json').read_text(encoding='utf-8-sig'))
original=json.loads((ROOT/'outputs/v22_full_style_20261001/content/narrative.json').read_text(encoding='utf-8-sig'))
changed_scripts=[i for i,(a,b) in enumerate(zip(original['slides'],narrative['slides']),1) if a['script']!=b['script']]
assert len(narrative['slides'])==23 and changed_scripts==[13,14],changed_scripts
han=sum(len(re.findall(r'[\u4e00-\u9fff]',s['script'])) for s in narrative['slides'] if not s['is_appendix'])
assert han==narrative['meta']['main_spoken_han_characters']
seconds=sum(s['seconds'] for s in narrative['slides'] if not s['is_appendix'])
assert seconds==narrative['meta']['planned_main_seconds']
assert han/300>12
assert .75*1000+.25*1200==1050
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','c':'http://schemas.openxmlformats.org/drawingml/2006/chart'}
def texts(blob):return [t.text or '' for t in ET.fromstring(blob).findall('.//a:t',ns)]
def charts(z):
    out=[]
    for part in sorted(p for p in z.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml',p)):
        c=ET.fromstring(z.read(part))
        out.append({'categories':[t.text for t in c.findall('.//c:cat//c:v',ns)],'values':[t.text for t in c.findall('.//c:val//c:v',ns)]})
    return out

records=[]
members={}
for key,title,folder,old_sha in styles:
    new=D/'deliverables'/f'臺灣房價預測_v22_{title}_r2.pptx'
    old=ROOT/'outputs'/folder/'deliverables'/f'臺灣房價預測_v22_{title}.pptx'
    assert sha(old)==old_sha
    v=json.loads((D/'build'/key/'final_r2.validation.json').read_text(encoding='utf-8'))
    assert v['finalSha256']==sha(new) and v['packageIntegrity']['status']=='pass'
    assert v['firstPartyImport']['passed'] and v['nativeQuantitativeCharts']['report']['passed']
    with zipfile.ZipFile(new) as z,zipfile.ZipFile(old) as o:
        assert z.testzip() is None
        slide_parts=[p for p in z.namelist() if re.fullmatch(r'ppt/slides/slide\d+\.xml',p)]
        assert len(slide_parts)==23
        changed=[]
        for i in range(1,24):
            if texts(z.read(f'ppt/slides/slide{i}.xml'))!=texts(o.read(f'ppt/slides/slide{i}.xml')):changed.append(i)
            note='\n'.join(ET.fromstring(z.read(f'ppt/notesSlides/notesSlide{i}.xml')).itertext())
            assert narrative['slides'][i-1]['script'] in note,(key,i)
        assert changed==[13,14],(key,changed)
        assert charts(z)==charts(o)
        table_count=sum(len(ET.fromstring(z.read(p)).findall('.//a:tbl',ns)) for p in slide_parts)
        assert table_count==(9 if key=='B' else 10)
        chart_count=len(charts(z))
        assert chart_count==(4 if key=='B' else 3)
    changed_pixels=[]
    for i in range(1,24):
        a=Image.open(ROOT/'outputs'/folder/'build/renders_r1'/f'slide-{i:02}.png').convert('RGB')
        b=Image.open(D/'build'/key/'renders_r2'/f'slide-{i:02}.png').convert('RGB')
        if a.size!=b.size or a.tobytes()!=b.tobytes():changed_pixels.append(i)
    assert changed_pixels==[13,14],(key,changed_pixels)
    members[f'{key}_{new.name}']=new
    records.append({'style':key,'path':str(new.relative_to(D)),'bytes':new.stat().st_size,'sha256':sha(new),
                    'changed_visible_text_slides':changed,'changed_rendered_slides':changed_pixels,
                    'other_21_slides_pixel_identical':True,'all_23_notes_match_revised_script':True,
                    'chart_categories_and_values_unchanged':True,'native_tables':table_count,'native_charts':chart_count,
                    'render_sha256':{f'slide-{i:02}':sha(D/'build'/key/'renders_r2'/f'slide-{i:02}.png') for i in range(1,24)}})

protected=json.loads((ROOT/'outputs/v22_beginner_20261001/qa/protected_before.json').read_text(encoding='utf-8'))['checks']
assert all(sha(ROOT/p)==v['sha256'] for p,v in protected.items())
qa=json.loads((D/'visual_review.json').read_text(encoding='utf-8'))
assert qa['all_six_changed_slides_individually_reviewed'] and qa['blocking_findings']==[]

for name in ['完整講稿.md','排練與問答.md']:members[name]=D/'content'/name
guide=f'''# v22 三款教學修訂版

本包的r2是簡報內容修訂，提交模型仍為v22。A白底學術、B深色科技、C暖色雜誌每款仍23頁。
本次只補強第13頁的75%／25%計算例與第14頁的每支獨立修正、修正後才取中位數。
所有算例是教學假設，不是測試交易。模型、CSV、Public分數均未更動，Private未知。

三款共用完整講稿，全部23頁備忘稿已同步。20頁主講有{han:,}個純漢字，
以250–300字／分估計{han/300:.2f}–{han/250:.2f}分鐘，逐頁規劃{seconds//60}:{seconds%60:02}。
這是字數估計，尚未真人計時。上台只選一款，附錄及問答不計主講時長。

完整講稿與排練問答已放入此包。已渲染檢查，未桌面PowerPoint／現場投影實測。
原r1三款保留，尚未替使用者選定風格。本包不含模型或交易資料，不是重訓備份。
'''
archive=D/'deliverables/v22_三種風格_教學修訂版.zip'
assert not archive.exists(),'Preserve existing delivery'
with zipfile.ZipFile(archive,'x',zipfile.ZIP_DEFLATED) as z:
    for name,path in members.items():z.write(path,name)
    z.writestr('請先閱讀.md',guide)
with zipfile.ZipFile(archive) as z:
    assert z.testzip() is None and len(z.namelist())==6
    for name,path in members.items():assert z.read(name)==path.read_bytes()
    assert z.read('請先閱讀.md').decode('utf-8')==guide

w,h,gap=640,360,12
sheet=Image.new('RGB',(3*w+gap*2,2*h+gap),(222,226,231))
for col,(key,*_) in enumerate(styles):
    for row,num in enumerate([13,14]):
        im=Image.open(D/'build'/key/'renders_r2'/f'slide-{num:02}.png').convert('RGB')
        sheet.paste(im.resize((w,h),Image.Resampling.LANCZOS),(col*(w+gap),row*(h+gap)))
preview=D/'deliverables/兩頁教學修訂預覽.png'
sheet.save(preview)
receipt={'checked_at_utc':datetime.now(timezone.utc).isoformat(),'styles':records,
         'main_spoken_han_characters':han,'planned_main_seconds':seconds,
         'estimated_minutes_at_250_to_300':[han/300,han/250],'human_timed_rehearsal':False,
         'script_changed_slides':changed_scripts,'new_ML_fits':0,'new_submission_CSV':False,
         'Private_score':None,'nine_protected_originals_unchanged':True,
         'all_three_r1_decks_unchanged':True,'PowerPoint_desktop_tested':False,
         'GitHub_published_this_revision':False,
         'archive':str(archive.relative_to(D)),'archive_bytes':archive.stat().st_size,
         'archive_sha256':sha(archive),'archive_member_count':6,'all_member_bytes_verified':True,
         'preview_sha256':sha(preview)}
(D/'delivery_receipt.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({k:v for k,v in receipt.items() if k!='styles'},ensure_ascii=False))
