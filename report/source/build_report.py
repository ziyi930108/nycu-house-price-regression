"""Remove section 6.4 while preserving every other document component."""
from pathlib import Path
from hashlib import sha256
from zipfile import ZipFile
from copy import deepcopy
from lxml import etree
import argparse,json
S=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--output-dir',type=Path,default=S.parent);a=ap.parse_args()
O=a.output_dir.resolve();O.mkdir(parents=True,exist_ok=True);(O/'qa').mkdir(exist_ok=True)
N='台灣房屋銷售價格預測挑戰賽_正式報告.docx';dest=O/N;assert not dest.exists(),'Refuse existing report'
PRIOR='adb7ecd04924ad64567e49a23ba1715b6e60442e6046c9731930e9c160c5b0c3'
assert sha256((S/'reference_report.docx').read_bytes()).hexdigest()==PRIOR
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
text=lambda e:''.join(e.xpath('.//w:t/text()',namespaces=ns))
with ZipFile(S/'reference_report.docx') as z:
    root=etree.fromstring(z.read('word/document.xml'));body=root.find('w:body',ns);nodes=list(body)
    starts=[i for i,n in enumerate(nodes) if text(n)=='6.4 公開排行榜成績'];assert len(starts)==1
    start=starts[0];end=next(i for i in range(start+1,len(nodes)) if text(nodes[i])=='七. 誤差分析與改進')
    removed=nodes[start:end];assert len(removed)==5
    assert [etree.QName(n).localname for n in removed]==['p','p','tbl','p','p']
    assert text(removed[1]).startswith('平台把test分成兩部分評分：')
    assert text(removed[3])==''
    assert text(removed[4]).startswith('公開MAE約199.58萬元')
    removed_text=[text(n) for n in removed]
    for n in removed:body.remove(n)
    changed=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
    with ZipFile(dest,'w') as out:
        for info in z.infolist():out.writestr(deepcopy(info),changed if info.filename=='word/document.xml' else z.read(info.filename))
md=(S/'reference_report.md').read_text(encoding='utf-8')
start=md.index('### 6.4 公開排行榜成績');end=md.index('## 七. 誤差分析與改進',start)
(O/'REPORT.md').write_text(md[:start]+md[end:],encoding='utf-8')
receipt={'status':'built_visual_review_pending','prior_sha256':PRIOR,'docx_sha256':sha256(dest.read_bytes()).hexdigest(),
 'removed_section':'6.4 公開排行榜成績','removed_nodes':removed_text,'changed_parts':['word/document.xml'],
 'tables':len(root.xpath('//w:tbl',namespaces=ns)),'images':len([n for n in z.namelist() if n.startswith('word/media/')])}
(O/'qa/build.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(receipt,ensure_ascii=False))
