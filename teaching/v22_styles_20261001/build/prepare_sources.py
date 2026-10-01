"""Create new style sources with two focused teaching improvements."""
from pathlib import Path
import hashlib,json,shutil
D=Path(__file__).resolve().parents[1]
ROOT=D.parents[1]
sources=[('A','v22_full_style_20261001'),('B','v22_dark_style_20261001'),('C','v22_warm_style_20261001')]
records=[]
for key,folder in sources:
    base=ROOT/'outputs'/folder
    target=D/'build'/key
    assert not (target/'build_deck.mjs').exists(), 'Preserve existing authoring source'
    original=base/'build/build_deck.mjs'
    s=original.read_text(encoding='utf-8-sig')
    old="const DIR=path.resolve(import.meta.dirname,'..');"
    assert old in s
    s=s.replace(old,"const DIR=path.resolve(import.meta.dirname,'../..');",1)
    s=s.replace("const BUILD=path.join(DIR,'build');",f"const BUILD=path.join(DIR,'build','{key}');",1)
    old_text="先還原總價，再按上表組分支；各支修正後，最後取中位數"
    assert old_text in s
    s=s.replace(old_text,"教學假設（萬元）：1,000×75% + 1,200×25% = 1,050")
    start=s.index('// 14 ')
    end=s.index('// 15 ',start)
    block="""// 14 Each branch is corrected independently; retain original base for spatial correction.
{const s=slide(14);tx(s,'01',72,153,116,70,46,{bold:true,color:C.green});tx(s,'成交參考',223,162,310,66,35,{bold:true});tx(s,'同建案、同地址、同道路的已知成交',223,236,980,61,32);tx(s,'再比較物件條件與時間趨勢，結合原估價及參考價',223,300,980,62,27,{color:C.muted});tx(s,'02',72,382,116,70,46,{bold:true,color:C.green});tx(s,'空間修正',223,391,310,66,35,{bold:true});tx(s,'參考支持較弱時才補充附近資訊，支持不足就回退',223,460,980,66,28);tx(s,'每條分支各自修正，保留三個總價，下一步才取中位數',72,543,1138,51,27,{bold:true,color:C.green});bottom(s,'價格只來自課程已知標籤；外部道路資料不提供成交答案',{size:25});}
"""
    s=s[:start]+block+s[end:]
    s=s.replace("narrativeSource:'../content/narrative.json'","narrativeSource:'../../content/narrative.json',branchExample:{illustrative:true,unit:'10kTWD',C_unit:1000,L_unit:1200,base:1050}")
    (target/'build_deck.mjs').write_text(s,encoding='utf-8')
    f=(base/'build/finalize_deck.mjs').read_text(encoding='utf-8-sig')
    assert "const DIR=path.resolve(import.meta.dirname,'..');" in f
    f=f.replace("const DIR=path.resolve(import.meta.dirname,'..');","const DIR=path.resolve(import.meta.dirname,'../..');",1)
    f=f.replace("const BUILD=path.join(DIR,'build'),OUT=",f"const BUILD=path.join(DIR,'build','{key}'),OUT=",1)
    f=f.replace("const revision=process.argv[2]||'r1';","const revision=process.argv[2]||'r2';",1)
    (target/'finalize_deck.mjs').write_text(f,encoding='utf-8')
    if key=='C':
        (target/'assets').mkdir()
        shutil.copyfile(base/'build/assets/taipei_markus_winkler.jpg',target/'assets/taipei_markus_winkler.jpg')
    records.append({'style':key,'original_folder':folder,'source_sha256':hashlib.sha256(original.read_bytes()).hexdigest(),'visible_changes':[13,14]})
(D/'build/source_derivation.json').write_text(json.dumps({'styles':records,'new_ML_fits':0,'original_files_modified':False},ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print('Three r2 source pairs prepared; waiting for revised shared narrative.')
