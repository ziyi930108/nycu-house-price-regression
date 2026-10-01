"""Export fixed-layout PDF backups from the verified v22 r2 slide renders.

Requires ReportLab, pypdf, Pillow and Poppler. Does not modify any source files.
PDF pages contain images; use the original PPTX for editable text and charts.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json, os, subprocess, time
from PIL import Image, ImageChops, ImageStat
from pypdf import PdfReader
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader

NAMES={'A':'白底學術','B':'深色科技','C':'暖色雜誌'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,value):
    with p.open('x',encoding='utf-8') as f:json.dump(value,f,ensure_ascii=False,indent=2);f.write('\n')
def rgb(path):
    with Image.open(path) as img:
        if 'A' in img.getbands() and img.getchannel('A').getextrema()!=(255,255):
            raise ValueError(f'Transparent pixels need an explicit compositing decision: {path}')
        return img.convert('RGB')

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--reference',type=Path,required=True,help='Frozen r2 teaching release with delivery_receipt and narrative')
    ap.add_argument('--render-root',type=Path,required=True,help='Directory containing A/B/C/renders_r2')
    ap.add_argument('--output',type=Path,required=True,help='New output directory, must not exist')
    ap.add_argument('--pdftoppm',type=Path,required=True)
    args=ap.parse_args()
    if os.path.lexists(args.output):ap.error('Output directory must not exist')
    ref,render_root,out=args.reference.resolve(),args.render_root.resolve(),args.output.resolve()
    poppler=args.pdftoppm.resolve()
    for source in [ref,render_root,poppler.parent]:
        if out.is_relative_to(source) or source.is_relative_to(out):raise ValueError('Output overlaps an input')
    assert poppler.is_file(),poppler
    receipt=json.loads((ref/'delivery_receipt.json').read_text(encoding='utf-8'))
    narrative=json.loads((ref/'content/narrative.json').read_text(encoding='utf-8'))
    assert len(narrative['slides'])==23 and sum(not s['is_appendix'] for s in narrative['slides'])==20
    frozen={}
    for e in receipt['styles']:
        key=e['style'];assert sha(ref/e['path'])==e['sha256']
        frozen[str(ref/e['path'])]=e['sha256']
        for n in range(1,24):
            p=render_root/key/'renders_r2'/f'slide-{n:02}.png'
            assert sha(p)==e['render_sha256'][f'slide-{n:02}'],p
            with Image.open(p) as img:assert img.size==(1600,900),p
            frozen[str(p)]=sha(p)
    out.mkdir(parents=True,exist_ok=False)
    (out/'deliverables').mkdir();(out/'renders').mkdir()
    started=time.perf_counter();records=[]
    try:
        for key,name in NAMES.items():
            pdf=out/'deliverables'/f'臺灣房價預測_v22_{name}_投影備份.pdf'
            c=canvas.Canvas(str(pdf),pagesize=(960,540),pageCompression=1,invariant=1)
            c.setTitle(f'臺灣房價預測 v22 - {name} - 投影備份')
            c.setAuthor('NYCU House Price Regression Project')
            c.setSubject('23頁固定畫面投影備份；第1-20頁主講，第21-23頁附錄。講稿另附。')
            for n,s in enumerate(narrative['slides'],1):
                source=render_root/key/'renders_r2'/f'slide-{n:02}.png'
                img=rgb(source)
                c.drawImage(ImageReader(img),0,0,width=960,height=540)
                bookmark=f'slide-{n:02}'
                c.bookmarkPage(bookmark)
                c.addOutlineEntry(f'{n:02} {s["title"]}',bookmark,level=0,closed=False)
                c.showPage()
            c.save()
            reader=PdfReader(pdf);assert len(reader.pages)==23 and len(reader.outline)==23
            dest=out/'renders'/key;dest.mkdir()
            command=[str(poppler),'-r','120','-png',str(pdf),str(dest/'slide')]
            start=time.perf_counter()
            result=subprocess.run(command,capture_output=True)
            (out/f'{key}_pdftoppm.stdout.txt').write_bytes(result.stdout)
            (out/f'{key}_pdftoppm.stderr.txt').write_bytes(result.stderr)
            assert result.returncode==0,result.stderr
            pages=[]
            for n,page in enumerate(reader.pages,1):
                source=render_root/key/'renders_r2'/f'slide-{n:02}.png'
                source_img=rgb(source)
                assert list(page.mediabox)==[0,0,960,540] and page.rotation==0
                assert len(page.images)==1 and not (page.extract_text() or '').strip()
                embedded=page.images[0].image.convert('RGB')
                assert embedded.size==source_img.size and embedded.tobytes()==source_img.tobytes()
                rendered=dest/f'slide-{n:02}.png'
                actual=rgb(rendered);assert actual.size==source_img.size
                diff=ImageChops.difference(source_img,actual)
                max_difference=max(v[1] for v in diff.getextrema())
                channels=[v.point(lambda x:255 if x else 0) for v in diff.split()]
                mask=ImageChops.lighter(ImageChops.lighter(channels[0],channels[1]),channels[2])
                changed_pixels=mask.histogram()[255]
                pages.append({'page':n,'title':narrative['slides'][n-1]['title'],'source_sha256':sha(source),
                              'embedded_pixels_exact':True,'render_sha256':sha(rendered),
                              'render_pixels_exact':changed_pixels==0,'changed_pixels':changed_pixels,
                              'max_channel_difference':max_difference,'mean_channel_difference':ImageStat.Stat(diff).mean})
            records.append({'style':key,'path':pdf.relative_to(out).as_posix(),'bytes':pdf.stat().st_size,'sha256':sha(pdf),
                            'pages':23,'main_pages':20,'appendix_pages':3,'image_dimensions':[1600,900],
                            'page_size_points':[960,540],'bookmark_count':23,'searchable_slide_text':False,
                            'editable_slide_objects':False,'speaker_notes_in_pdf':False,'render_command':command,
                            'poppler_exit_code':result.returncode,'render_and_comparison_seconds':time.perf_counter()-start,
                            'all_embedded_images_exact':True,'all_render_pixels_exact':all(p['render_pixels_exact'] for p in pages),
                            'page_checks':pages})
            print(json.dumps({k:records[-1][k] for k in ['style','bytes','sha256','all_embedded_images_exact','all_render_pixels_exact']}),flush=True)
        assert all(sha(Path(p))==h for p,h in frozen.items())
        save(out/'delivery_receipt.json',{'created_at_utc':datetime.now(timezone.utc).isoformat(),'styles':records,
             'elapsed_seconds':time.perf_counter()-started,'originals_unchanged':True,'reference':str(ref),
             'reference_receipt_sha256':sha(ref/'delivery_receipt.json'),'render_root':str(render_root),
             'method':'Lossless RGB images placed full-page; original notes remain in PPTX and separate script',
             'build_script_sha256':sha(Path(__file__)),'new_ML_fits':0,'new_submission_CSV':False,
             'Private_score':None,'physical_projection_tested':False})
    except Exception as exc:
        save(out/'failure.json',{'error':str(exc),'completed_styles':records,'elapsed_seconds':time.perf_counter()-started})
        raise

if __name__=='__main__':main()
