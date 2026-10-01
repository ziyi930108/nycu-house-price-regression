from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as E
from datetime import datetime,timezone
import json,hashlib,shutil,re
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=HERE.parent/'v22_teaching_refinement_20261001';OUT=HERE/'deliverables'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def js(p,d):write(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def txt(b):return '\n'.join(x.text or '' for x in E.fromstring(b).findall('.//a:t',ns))
plan=json.loads((HERE/'content/notes_plan.json').read_text(encoding='utf-8'));result=[]
old_receipt=json.loads((OLD/'delivery_receipt.json').read_text(encoding='utf-8'))
changed={f'ppt/notesSlides/notesSlide{i}.xml' for i in range(1,21)}
for e in plan['styles']:
 key=e['style'];old=OLD/'deliverables'/e['source_file'];new=OUT/e['source_file'].replace('_r2.pptx','_口語講稿版.pptx')
 assert sha(old)==e['source_sha256']
 with ZipFile(old) as o,ZipFile(new) as n:
  assert n.testzip() is None and set(n.namelist())==set(o.namelist())
  differences={p for p in n.namelist() if n.read(p)!=o.read(p)};assert differences==changed,(key,differences^changed)
  for i in range(1,24):assert txt(n.read(f'ppt/notesSlides/notesSlide{i}.xml'))==e['notes'][i-1],(key,i)
  tables=sum(len(E.fromstring(n.read(f'ppt/slides/slide{i}.xml')).findall('.//a:tbl',ns)) for i in range(1,24))
  charts=[p for p in n.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml',p)]
  workbooks=[p for p in n.namelist() if p.startswith('ppt/embeddings/') and p.endswith('.xlsx')]
  assert len(charts)==len(workbooks)==(4 if key=='B' else 3)
  assert tables==(9 if key=='B' else 10)
 renders={f'slide-{i:02}':sha(HERE/'build'/key/'renders'/f'slide-{i:02}.png') for i in range(1,24)}
 source_entry=next(x for x in old_receipt['styles'] if x['style']==key)
 assert renders==source_entry['render_sha256'],key
 result.append({'style':key,'path':new.relative_to(HERE).as_posix(),'sha256':sha(new),'bytes':new.stat().st_size,'slides':23,'changed_notes':20,'all_23_notes_match':True,'all_other_package_parts_identical':True,'all_23_render_bytes_identical':True,'native_tables':tables,'native_charts':len(charts),'unchanged_embedded_workbooks':len(workbooks),'render_sha256':renders})
protected=json.loads((ROOT/'outputs/github_publish_v22_readability_20261001/protected_before.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/p)==h for p,h in protected.items())
pdfs=[]
for p in sorted((ROOT/'outputs/v22_projection_pdf_20261001/final/deliverables').glob('*.pdf')):
 q=OUT/p.name;assert not q.exists();shutil.copy2(p,q);assert sha(p)==sha(q);pdfs.append({'path':q.relative_to(HERE).as_posix(),'sha256':sha(q),'unchanged_projection_backup':True})
assert len(pdfs)==3
for name in ('完整口語講稿.md','排練與問答.md','上台口述提示.md'):
 q=OUT/name;assert not q.exists();shutil.copy2(HERE/'content'/name,q)
write(OUT/'先讀我.md','''# v22 口語講稿版上台包

這三款都是同一個v22模型、相同20頁主講加3頁附錄。選一款PPTX報告即可，無須把三款都講一遍。尚未指定風格時，可先用白底學術版排練。

- PPTX：主講備忘稿已換成白話口語稿，原畫面與附錄保持。
- 完整口語講稿：從第1頁到第20頁完整練一次，說清每一步為什麼做。
- 上台口述提示：每頁一句提醒，忘記下一句時找回思路。
- 排練與問答：計時方法與16題問答，問答不算入主講時間。
- PDF：保留原投影畫面，沒有講者備忘稿，供字型或軟體顯示異常時使用。

主講4,981純漢字，按250–300字／分鐘估計16.60–19.92分鐘；逐頁規劃17分20秒。尚未真人計時，請實際完整排練。公開分數口述約兩百萬元，再說明這是平均絕對誤差，Private仍未知。

此版只改講解，沒有新訓練、分數或提交CSV。v22 Public仍為1,998,568.5706元，Private未知。
''')
archive=HERE/'v22_三種風格_口語上台包.zip'
with ZipFile(archive,'x',compression=ZIP_DEFLATED,compresslevel=6) as z:
 for p in sorted(OUT.iterdir()):z.write(p,p.name)
with ZipFile(archive) as z:
 assert len(z.namelist())==10 and z.testzip() is None
 assert all(z.read(p.name)==p.read_bytes() for p in OUT.iterdir())
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'styles':result,'unchanged_pdf_backups':pdfs,'all_69_visual_render_bytes_identical':True,'nine_protected_identities_unchanged':True,'protected_sha256':protected,'only_edited_pptx_parts':'notesSlides 1 through 20','archive':{'path':archive.relative_to(HERE).as_posix(),'files':10,'bytes':archive.stat().st_size,'sha256':sha(archive),'CRC_and_member_bytes_verified':True},'main_han':4981,'planned_main_seconds':1040,'human_rehearsal_measured':False,'PowerPoint_desktop_tested':False,'Private_score':None,'new_ML_fits':0,'attempt1_failure':'Artifact export omitted embedded workbooks. Original package preserved with only authored notes replacements; same finalization gates passed.'}
js(HERE/'delivery_receipt.json',receipt)
print(json.dumps({'styles':len(result),'render_bytes_identical':69,'archive':receipt['archive']},ensure_ascii=False))
