from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime,timezone
import hashlib,json,shutil
ROOT=Path(__file__).resolve().parents[2];WORK=Path(__file__).resolve().parent
OUT=ROOT/'deliverables/v22_審查資料夾_20261001'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,d):
    with p.open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
checks=json.loads((WORK/'browser_verification.json').read_text('utf-8'));assert checks['passed'] and checks['video_metadata_and_seek_tested']
video=json.loads((WORK/'video_final/receipt.json').read_text('utf-8'));assert video['status']=='completed'
protected=json.loads((ROOT/'outputs/github_publish_v22_delivery_coaching_20261001/protected_before.json').read_text('utf-8'))
assert all(sha(ROOT/p)==h for p,h in protected.items())
for p in [WORK/'browser_verification.json',WORK/'finish_review.py',WORK/'verify_review.cjs',WORK/'package_review.py']:
    q=OUT/'04_程式與證據/本次製作程式'/p.name;assert not q.exists();shutil.copy2(p,q)
files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)} for p in sorted(OUT.rglob('*')) if p.is_file()]
manifest={'created_at_utc':datetime.now(timezone.utc).isoformat(),'edition':'v22_review_20261001','purpose':'User review folder with optimized notes and narrated video','selected_model':'v22','private_score':None,'main_han':4065,'planned_speech_seconds':885,'synthetic_video_seconds':900,'real_human_delivery_measured':False,'visible_slides_unchanged':True,'changed_notes_pages':[12,13],'protected_originals_unchanged':True,'new_model_fits':0,'github_new_review_edition_published':False,'files_exclude_this_manifest':True,'files':files}
save(OUT/'manifest.json',manifest)
archive=ROOT/'deliverables/v22_審查與影片_20261001.zip'
with ZipFile(archive,'x',ZIP_DEFLATED) as z:
    for p in sorted(OUT.rglob('*')):
        if p.is_file():z.write(p,OUT.name+'/'+p.relative_to(OUT).as_posix())
with ZipFile(archive) as z:
    assert z.testzip() is None
    for p in (x for x in OUT.rglob('*') if x.is_file()):assert hashlib.sha256(z.read(OUT.name+'/'+p.relative_to(OUT).as_posix())).hexdigest()==sha(p)
receipt={'created_at_utc':datetime.now(timezone.utc).isoformat(),'folder':OUT.relative_to(ROOT).as_posix(),'archive':archive.relative_to(ROOT).as_posix(),'archive_bytes':archive.stat().st_size,'archive_sha256':sha(archive),'files_in_folder':len(files)+1,'manifest_sha256':sha(OUT/'manifest.json'),'zip_crc_and_members_verified':True,'browser_checks_passed':True,'original_protected_hashes_unchanged':True,'pptx_build_seconds':json.loads((WORK/'build/execution.json').read_text('utf-8'))['elapsed_seconds'],'video_sha256':video['main_mp4']['sha256'],'video_seconds':900,'video_full_decode_passed':True,'human_audio_listening_performed':False,'human_rehearsal_measured':False,'new_model_fits':0,'private_score':None}
save(WORK/'delivery_receipt.json',receipt)
print(json.dumps(receipt,ensure_ascii=False))
