"""Verify notes-only updates, synchronize rehearsal, package the 15-minute edition."""
from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from xml.etree import ElementTree as E
from datetime import datetime,timezone
import json,hashlib,re,shutil
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1];OLD=ROOT/'outputs/v22_teaching_refinement_20261001';OUT=HERE/'deliverables'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,s):
 with p.open('x',encoding='utf-8') as f:f.write(s)
def save(p,d):write(p,json.dumps(d,ensure_ascii=False,indent=2)+'\n')
plan=json.loads((HERE/'content/notes_plan.json').read_text(encoding='utf-8'));narrative=json.loads((HERE/'content/narrative.json').read_text(encoding='utf-8'))
old_receipt=json.loads((OLD/'delivery_receipt.json').read_text(encoding='utf-8'))
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
def txt(b):return '\n'.join(t.text or '' for t in E.fromstring(b).findall('.//a:t',ns))
changed={f'ppt/notesSlides/notesSlide{i}.xml' for i in range(1,21)}
results=[]
for entry in plan['styles']:
 key=entry['style'];original=OLD/'deliverables'/entry['source_file'];final=OUT/entry['source_file'].replace('_r2.pptx','_15分鐘版.pptx')
 assert sha(original)==entry['source_sha256']
 with ZipFile(original) as a,ZipFile(final) as b:
  assert b.testzip() is None and set(a.namelist())==set(b.namelist())
  assert {p for p in b.namelist() if a.read(p)!=b.read(p)}==changed
  for i in range(1,24):assert txt(b.read(f'ppt/notesSlides/notesSlide{i}.xml'))==entry['notes'][i-1]
  tables=sum(len(E.fromstring(b.read(f'ppt/slides/slide{i}.xml')).findall('.//a:tbl',ns)) for i in range(1,24))
  charts=[p for p in b.namelist() if re.fullmatch(r'ppt/(?:slides/)?charts/chart\d+\.xml',p)]
  workbooks=[p for p in b.namelist() if p.startswith('ppt/embeddings/') and p.endswith('.xlsx')]
  assert len(charts)==len(workbooks)==(4 if key=='B' else 3)
  assert tables==(9 if key=='B' else 10)
 renders={f'slide-{i:02}':sha(HERE/'build'/key/'renders'/f'slide-{i:02}.png') for i in range(1,24)}
 assert renders==next(s for s in old_receipt['styles'] if s['style']==key)['render_sha256']
 results.append({'style':key,'path':final.relative_to(HERE).as_posix(),'sha256':sha(final),'slides':23,'notes_matched':23,'only_changed_parts':sorted(changed),'other_parts_identical':True,'native_tables':tables,'native_charts':len(charts),'original_workbooks_preserved':len(workbooks),'render_sha256':renders})
protected=json.loads((ROOT/'outputs/github_publish_v22_rehearsal_tool_20261001/protected_before.json').read_text(encoding='utf-8'))
assert all(sha(ROOT/p)==h for p,h in protected.items())
copies=[]
sources=list((ROOT/'outputs/v22_projection_pdf_20261001/final/deliverables').glob('*.pdf'))
sources += [HERE/'content'/name for name in ('完整口語講稿.md','排練與問答.md','上台口述提示.md')]
sources += [ROOT/'outputs/v22_defense_cards_20261001/final'/('臺灣房價預測_v22_答辯提示卡'+suffix) for suffix in ('.pdf','.md')]
assert len(sources)==8
for p in sources:
 q=OUT/p.name;assert not q.exists();shutil.copy2(p,q);assert sha(p)==sha(q);copies.append({'source':p.relative_to(ROOT).as_posix(),'file':q.name,'sha256':sha(q)})
template=(ROOT/'outputs/v22_rehearsal_tool_20261001/template.html').read_text(encoding='utf-8')
template=template.replace('v22 上台排練','v22 約15分鐘上台排練').replace('原講稿規劃 17:20，尚無本人排練結果。','本版目標約15分鐘，尚無本人排練結果。先自然講完，再調整接近15分鐘。')
template=template.replace('minimum_seconds:720,','minimum_seconds:720,target_seconds:900,')
template=template.replace("r.total_seconds>=720?'計時已到 12 分鐘，仍需確認每頁內容都有說清楚。':'計時尚未到 12 分鐘，先檢查是否漏了方法理由與算例，不靠等待補時間。'", "r.total_seconds>960?'已超過16分鐘，可檢查重複轉場，保留核心例子與限制。':r.total_seconds>=840?'計時接近15分鐘，仍需確認每頁內容都有說清楚。':r.total_seconds>=720?'已滿12分鐘，但離約15分鐘的目標偏短，檢查有沒有漏講設計理由。':'計時尚未到12分鐘，檢查是否漏了方法理由與算例，不靠等待補時間。'")
payload={'source_sha256':sha(HERE/'content/narrative.json'),'slides':[{k:s[k] for k in ('number','title','takeaway','script','seconds')} for s in narrative['slides'][:20]]}
assert len(payload['slides'])==20 and template.count('__DECK_JSON__')==1
write(HERE/'rehearsal_template.html',template)
write(OUT/'v22-15min-rehearsal.html',template.replace('__DECK_JSON__',json.dumps(payload,ensure_ascii=False).replace('<','\\u003c')))
count=narrative['meta']['main_spoken_han_characters'];seconds=narrative['meta']['planned_main_seconds']
write(OUT/'先讀我.md',f'''# v22 約15分鐘上台包

這三款都是同一個v22模型，20頁主講加3頁附錄。請選一款PPTX報告，不必連播三份。可先以白底學術版排練，再按喜好選風格。

1. 開啟所選PPTX，講者備忘稿已同步本次精簡口語稿。
2. 用v22-15min-rehearsal.html開啟離線排練，先看一句提示再用自己的話說，卡住才展開全文。記錄實際時間及卡住的地方。
3. 完整口語講稿與每頁口述提示方便複習。排練與問答、四頁答辯卡留作問答準備，不計主講時長。
4. 對應風格的PDF是固定畫面備份，沒有講者備忘稿；上台前在實際電腦試播。

主講{count:,}純漢字，以每分鐘275字估計{count/275:.2f}分鐘，逐頁規劃{seconds//60}:{seconds%60:02}。250至300字／分的估計範圍為{count/300:.2f}至{count/250:.2f}分鐘，並非每個人自然說話都會剛好15分鐘。目標約15分鐘，至少12分鐘，不靠附錄／問答／等待補時。尚未本人計時。

本版刪減口語重複，保留完整流程、每步理由與重要例子。Public仍為1,998,568.5706元，Private未知，模型仍為唯一選定的v22。這是上台成品包，不含原始交易或模型權重。
''')
archive=HERE/'v22_約15分鐘_完整上台包.zip'
with ZipFile(archive,'x',ZIP_DEFLATED) as z:
 for p in sorted(OUT.iterdir()):z.write(p,p.name)
with ZipFile(archive) as z:
 assert len(z.namelist())==13 and z.testzip() is None
 assert all(z.read(p.name)==p.read_bytes() for p in OUT.iterdir())
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'styles':results,'main_han':count,'planned_main_seconds':seconds,'estimated_minutes_at_275':count/275,'main_text_reduction_from_4981':4981-count,'all_69_render_bytes_identical':True,'protected_hashes_unchanged':True,'unchanged_copies':copies,'rehearsal_script_source_sha256':payload['source_sha256'],'archive':{'path':archive.name,'files':13,'bytes':archive.stat().st_size,'sha256':sha(archive),'all_members_verified':True},'human_rehearsal_measured':False,'PowerPoint_desktop_tested':False,'new_ML_fits':0,'Private_score':None}
save(HERE/'delivery_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
