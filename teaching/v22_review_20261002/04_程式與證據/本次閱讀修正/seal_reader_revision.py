from pathlib import Path
from zipfile import ZipFile,ZIP_DEFLATED
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OLD=ROOT/'deliverables/v22_審查資料夾_20261002';OUT=ROOT/'deliverables/v22_審查資料夾_20261002_r2'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
assert not (OUT/'manifest.json').exists()
checks={name:json.loads((WORK/name).read_text('utf-8-sig')) for name in ('report_integration.json','browser/browser_receipt.json','markdown_links.json','powershell_guide_syntax.json')}
assert checks['browser/browser_receipt.json']['passed'] and checks['markdown_links.json']['passed']
assert all(not x['errors'] for x in checks['powershell_guide_syntax.json']['results'])
original=json.loads((OLD/'manifest.json').read_text('utf-8'))
assert all(sha(OLD/x['path'])==x['sha256'] for x in original['files'])
protected=json.loads((ROOT/'outputs/github_publish_v22_delivery_coaching_20261001/protected_before.json').read_text('utf-8'))
assert all(sha(ROOT/p)==h for p,h in protected.items())
for folder in ('01_上台簡報','05_講解影片'):
    assert all(sha(p)==sha(OUT/p.relative_to(OLD)) for p in (OLD/folder).rglob('*') if p.is_file())
assert sha(OUT/'04_程式與證據/submission_v22_trial.csv')=='6490e2da78d4bf8173a27444de673a020152878f78e7ebec54d4fdc9bddfae35'
assert sha(OUT/'04_程式與證據/本次內容/narrative.json')==sha(OLD/'04_程式與證據/本次內容/narrative.json')
assert sha(OUT/'02_講稿與排練/v22-review-rehearsal.html')==sha(OLD/'02_講稿與排練/v22-review-rehearsal.html')
code=OUT/'04_程式與證據/正式報告字型修正'
source=ROOT/'outputs/v22_reader_review_20261002/pdf_revision/original_source_verification.json'
shutil.copy2(source,code/source.name)
p=code/'README.md';p.write_text(p.read_text('utf-8')+'\n\n另已使用修正版builder直接讀原程式包REPORT.md，新的11頁全文、繪圖stream及字型資源與本修正版完全相同，詳見[原來源重建核對](original_source_verification.json)。因此也可將上例`--input`指定為原程式包的REPORT.md；原檔不必改寫。這次比較沒有重渲全報告，也未重跑模型。\n',encoding='utf-8')
add=OUT/'04_程式與證據/本次閱讀修正';add.mkdir()
for name in ('prepare_reader_revision.py','integrate_report.py','seal_reader_revision.py','report_integration.json','powershell_guide_syntax.json'):
    shutil.copy2(WORK/name,add/name)
changes=[]
for p in sorted(OUT.rglob('*')):
    if p.is_file():
        rel=p.relative_to(OUT);old=OLD/rel
        if not old.exists() or sha(old)!=sha(p):changes.append({'path':rel.as_posix(),'old_sha256':sha(old) if old.exists() else None,'new_sha256':sha(p)})
save(add/'revision_changes.json',{'previous_edition':'v22_review_20261002','new_edition':'v22_review_20261002_r2',
    'previous_manifest_sha256':sha(OLD/'manifest.json'),'changes':changes,
    'pptx_and_video_unchanged':True,'narrative_unchanged':True,'new_model_fits':0,'private_score':None})
files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'edition':'v22_review_20261002_r2',
    'selected_model':'v22','public_mae_twd':1998568.5706,'private_score':None,'main_han':4065,'planned_seconds':885,
    'synthetic_video_seconds':900,'real_human_delivery_measured':False,'pptx_video_and_narrative_unchanged':True,
    'report_change':'ASCII minus visible on page 10; other ten pages have identical pixels',
    'new_model_fits':0,'new_github_publication':False,'files_exclude_this_manifest':True,'files':files}
save(OUT/'manifest.json',manifest)
archive=ROOT/'deliverables/v22_審查與影片_20261002_r2.zip'
with ZipFile(archive,'x',ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
with ZipFile(archive) as z:
    assert z.testzip() is None
    for p in OUT.rglob('*'):
        if p.is_file():assert hashlib.sha256(z.read(OUT.name+'/'+p.relative_to(OUT).as_posix())).hexdigest()==sha(p)
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'folder':OUT.relative_to(ROOT).as_posix(),
    'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),'archive_bytes':archive.stat().st_size,
    'files_in_folder':len(files)+1,'manifest_sha256':sha(OUT/'manifest.json'),'zip_crc_and_member_sha_passed':True,
    'old_review_and_nine_protected_files_unchanged':True,'pptx_video_and_narrative_unchanged':True,
    'new_pdf_sha256':sha(OUT/'03_正式報告/臺灣房價預測_v22_正式報告.pdf'),
    'new_model_fits':0,'new_github_publication':False,'private_score':None}
save(WORK/'delivery_receipt.json',receipt);print(json.dumps(receipt,ensure_ascii=False))
