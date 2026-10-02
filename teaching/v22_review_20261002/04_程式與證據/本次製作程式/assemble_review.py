from pathlib import Path
from datetime import datetime, timezone
import json, shutil, hashlib, html
ROOT=Path(__file__).resolve().parents[2]
WORK=Path(__file__).resolve().parent
OUT=ROOT/'deliverables/v22_審查資料夾_20261001'
OUT.mkdir(exist_ok=False)
for name in ('01_上台簡報','02_講稿與排練','03_正式報告','04_程式與證據','05_講解影片','assets'):
    (OUT/name).mkdir()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
copies=[]
def cp(source,dest):
    source=Path(source);dest=OUT/dest
    dest.parent.mkdir(parents=True,exist_ok=True)
    assert not dest.exists()
    shutil.copy2(source,dest)
    assert sha(source)==sha(dest)
    copies.append({'source':source.relative_to(ROOT).as_posix(),'path':dest.relative_to(OUT).as_posix(),'sha256':sha(dest)})
def write(path,s):
    with (OUT/path).open('x',encoding='utf-8') as f:f.write(s)
for p in sorted((WORK/'deliverables').glob('*.pptx')):cp(p,'01_上台簡報/'+p.name)
assert len(list((OUT/'01_上台簡報').glob('*.pptx')))==3
for p in (ROOT/'outputs/v22_15min_20261001/deliverables').glob('*投影備份.pdf'):cp(p,'01_上台簡報/'+p.name)
for name in ('完整口語講稿.md','上台口述提示.md','排練與問答.md','v22-review-rehearsal.html'):
    cp(WORK/'content'/name,'02_講稿與排練/'+name)
for p in (ROOT/'outputs/v22_defense_cards_20261001/final').glob('臺灣房價預測_v22_答辯提示卡.*'):
    if p.suffix in ('.md','.pdf'):cp(p,'02_講稿與排練/'+p.name)
cp(ROOT/'deliverables/v22_github/docs/臺灣房價預測_v22_正式報告.pdf','03_正式報告/臺灣房價預測_v22_正式報告.pdf')
cp(ROOT/'deliverables/v22_github_release_20261001_delivery_coaching.zip','04_程式與證據/v22_程式與研究紀錄.zip')
cp(ROOT/'deliverables/v22_github_release_20261001_delivery_coaching.zip.json','04_程式與證據/原程式包身分.json')
cp(ROOT/'deliverables/v22_github/evidence/reproduction/integration_verified.json','04_程式與證據/保存模型推論重現收據.json')
cp(ROOT/'outputs/v22_submission_focus_20260928/user_selection.json','04_程式與證據/v22選定紀錄.json')
cp(ROOT/'outputs/submission_v22_trial.csv','04_程式與證據/submission_v22_trial.csv')
for name in ('narrative.json','notes_plan.json','approved_changes.json','timing.json'):
    cp(WORK/'content'/name,'04_程式與證據/本次內容/'+name)
for p in (WORK/'build_source').iterdir():
    if p.is_file():cp(p,'04_程式與證據/本次製作程式/'+p.name)
for name in ('prepare_review.py','assemble_review.py'):
    cp(WORK/name,'04_程式與證據/本次製作程式/'+name)
cp(WORK/'script_proposal.json','04_程式與證據/本次內容/script_proposal.json')
for k in ('A','B','C'):
    cp(WORK/'build'/k/'renders/slide-01.png',f'assets/{k}-cover.png')
    for n in (12,13):cp(WORK/'build'/k/'renders'/f'slide-{n:02}.png',f'assets/{k}-{n}.png')
for name in ('00_審查指南.md','審查意見.md'):cp(WORK/'guide'/name,name)
write('04_程式與證據/先讀我.md','''# 程式與證據

`v22_程式與研究紀錄.zip`是已發布私人GitHub的完整程式快照，commit為d747e98fef5842c293c2caf65c1f1aea5bd6e56b。解壓後先讀README.md、REPORT.md與docs/REPRODUCIBILITY.md，再看src/predict_v22.py。各輪歷史保留在research與evidence，無須按版本順序閱讀。

保存模型推論重現收據記錄原始test加17個保存artifact重跑47,073筆，CSV與v22一致。這份程式包不含原始交易表及約1.45GB模型與狀態；全新安裝、從零重訓全鏈尚未驗證。附上的submission_v22_trial.csv為原選定檔的逐位複製，沒有產生新價格。

本次內容和製作程式是新的審查版講稿與備忘稿來源，與原程式ZIP分開保存。notes builder參數化runtime/source/output；prepare_review與assemble_review記錄本工作區整合方式。這不代表新的審查稿已推上GitHub。模型及已發布來源保持。
''')
n=json.loads((WORK/'content/narrative.json').read_text('utf-8'))
c=json.loads((WORK/'content/approved_changes.json').read_text('utf-8'))
esc=html.escape
def paras(s):return ''.join('<p>'+esc(x)+'</p>' for x in s.split('\n\n'))
styles=[]
for key,label,desc in [('A','白底學術','先看這款，適合教室與列印。'),('B','深色科技','深底配亮字，可比較投影閱讀感。'),('C','暖色雜誌','暖色版面，內容和主講順序相同。')]:
    ppt=next((OUT/'01_上台簡報').glob('*'+label+'*.pptx')).name
    pdf=next((OUT/'01_上台簡報').glob('*'+label+'*.pdf')).name
    styles.append(f'<article><img src="assets/{key}-cover.png" alt="{label}簡報封面"><h3>{label}</h3><p>{desc}</p><a href="01_上台簡報/{ppt}">開啟 PPTX</a>　<a href="01_上台簡報/{pdf}">投影 PDF</a></article>')
notes=[]
for s in n['slides']:
    label='附錄，不計主講' if s['is_appendix'] else f"配速參考 {s['seconds']} 秒"
    notes.append(f'<details id="page-{s["number"]}" class="page"><summary><span>{s["number"]:02}</span> {esc(s["title"])} <small>{label}</small></summary><div class="note"><p class="takeaway">{esc(s["takeaway"])}</p><div data-script="{s["number"]}">{paras(s["script"])}</div><p class="reason">設計理由：{esc(s["why"])}</p></div></details>')
changes=[]
for x in c['changes']:
    changes.append(f'<article class="change"><h3>第{x["page"]}頁　{esc(x["title"])}</h3><p>{esc(x["reason"])}</p><div class="compare"><div><h4>修改前</h4>{paras(x["before"])}</div><div><h4>本次講稿</h4>{paras(x["after"])}</div></div></article>')
metadata={'edition':n['meta']['edition'],'narrative_sha256':sha(WORK/'content/narrative.json'),'public_mae_twd':1998568.5706,'private_score':None,'planned_seconds':885,'human_rehearsal_measured':False}
template='''<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>v22 簡報審查</title><style>
:root{color-scheme:light;--ink:#182b3b;--muted:#566775;--blue:#174f83;--line:#cdd6da;--paper:#f5f4ef}*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);font-family:"Microsoft JhengHei",system-ui,sans-serif;line-height:1.8}main{max-width:1180px;margin:auto;padding:48px 32px}header{border-top:6px solid var(--ink);padding:24px 0 30px}.eyebrow{font-size:14px;letter-spacing:.1em;color:var(--muted)}h1{font-size:clamp(32px,5vw,56px);line-height:1.3;margin:16px 0}h2{font-size:28px;margin:0 0 18px}h3{font-size:21px;margin:12px 0 6px}p{margin:8px 0 16px}a{color:var(--blue);text-underline-offset:4px}nav{display:flex;gap:22px;flex-wrap:wrap;border-top:1px solid var(--line);border-bottom:1px solid var(--line);padding:16px 0}section{padding:38px 0;border-bottom:1px solid var(--line)}.lead{font-size:21px;max-width:850px}.meta{font-size:15px;color:var(--muted)}.styles{display:grid;grid-template-columns:repeat(3,1fr);gap:24px}.styles img{width:100%;display:block;border:1px solid var(--line)}.styles p{font-size:15px}.split{display:grid;grid-template-columns:1fr 1fr;gap:36px}.compare{display:grid;grid-template-columns:1fr 1fr;gap:28px;font-size:16px}.compare>div:first-child{color:var(--muted)}.change{margin:25px 0 40px}.change h4{border-bottom:2px solid var(--line);padding-bottom:8px}.page{border-top:1px solid var(--line)}summary{padding:15px 0;cursor:pointer;font-size:19px}.page summary span{font-variant-numeric:tabular-nums;color:var(--blue);margin-right:12px}.page small{display:block;margin-left:40px;color:var(--muted);font-size:14px}.note{max-width:900px;margin:0 0 24px 40px;font-size:19px}.takeaway{font-weight:bold}.reason{border-left:3px solid var(--blue);padding-left:15px;font-size:16px}.button,button{display:inline-block;padding:10px 18px;background:var(--ink);color:white;border:0;border-radius:3px;font:inherit;cursor:pointer;text-decoration:none}button.secondary{background:transparent;color:var(--blue);border:1px solid var(--line);margin-right:8px}.formgrid{display:grid;grid-template-columns:1fr 1fr;gap:20px}label{display:block;font-weight:bold}input,select,textarea{display:block;width:100%;font:inherit;padding:10px;border:1px solid #87949e;border-radius:3px;background:white;color:var(--ink)}textarea{min-height:125px;resize:vertical}input:focus-visible,select:focus-visible,textarea:focus-visible,a:focus-visible,button:focus-visible,summary:focus-visible{outline:3px solid #cd751b;outline-offset:3px}#status{min-height:28px}footer{padding:24px 0;color:var(--muted);font-size:14px}video{display:block;width:100%;max-height:690px;background:#122334}figure{margin:0}figcaption{font-size:14px;color:var(--muted)}@media(max-width:700px){main{padding:24px 18px}.styles,.split,.compare,.formgrid{grid-template-columns:1fr}.styles{gap:30px}.note{margin-left:8px;font-size:18px}h2{font-size:25px}nav{gap:14px}section{padding:28px 0}}@media print{nav,button,form,video{display:none}main{max-width:none}details{break-inside:avoid}.styles{grid-template-columns:repeat(3,1fr)}}
</style></head><body><main><header><div class="eyebrow">NYCU · 房屋總價回歸 · 2026.10.01</div><h1>先看懂，再講清楚。</h1><p class="lead">這份審查版把第 12、13 頁的說明寫得更直接。三款簡報、講稿與旁白都以你選定的 v22 為主線。</p><p class="meta">20 頁主講 ＋ 3 頁附錄　｜　4,065 個漢字　｜　主講規劃 14:45，尚未真人計時</p><a class="button" href="#slides">從簡報開始</a>　<a href="00_審查指南.md">閱讀審查指南</a></header>
<nav aria-label="閱讀順序"><a href="#slides">簡報</a><a href="#script">逐頁講稿</a><a href="#changes">這次修改</a><a href="#video">講解影片</a><a href="#report">報告與程式</a><a href="#feedback">審查意見</a></nav>
<section id="slides"><h2>選一款繼續看</h2><p>三款內容和講稿相同。PPTX 含最新講者備忘稿，PDF 保留相同投影畫面。</p><div class="styles">__STYLES__</div></section>
<section id="script"><h2>逐頁口語講稿</h2><p>可先看第 12 至 15 頁的連續流程，再完整練第 1 至 20 頁。<a href="02_講稿與排練/v22-review-rehearsal.html">開啟離線排練計時</a></p><p><button class="secondary" id="openAll" type="button">展開全部</button><button class="secondary" id="closeAll" type="button">收合全部</button></p>__NOTES__</section>
<section id="changes"><h2>本次修改的兩段</h2><p>第 12 頁說清楚價格起點與倍率，第 13 頁按三路表格講解。投影片畫面、正式報告及模型價格保持。下方保留修改前後，方便直接比較。</p>__CHANGES__</section>
<section id="video"><h2>投影片講解影片</h2><div id="videoContent">影片會在簡報檢查完成後，依本版講稿製作。</div></section>
<section id="report"><h2>報告與程式</h2><div class="split"><div><h3>正式研究報告</h3><p>包含資料清理、模型設計、三個年度回測與限制。</p><p><a href="03_正式報告/臺灣房價預測_v22_正式報告.pdf">開啟完整 PDF 報告</a></p><p>Public MAE 為 1,998,568.5706 元，Private 仍未知。這是平均絕對金額誤差，不是準確率。</p></div><div><h3>程式與重現證據</h3><p><a href="04_程式與證據/v22_程式與研究紀錄.zip">下載程式與研究紀錄 ZIP</a>　<a href="04_程式與證據/先讀我.md">先看重現範圍</a></p><p>已用原始 test 與保存模型重跑 47,073 筆並得到相同 CSV。全新環境安裝及從零重訓尚未驗證，程式包不包含原始資料與大型模型。</p><p><a href="04_程式與證據/submission_v22_trial.csv">原選定 v22 提交檔</a></p></div></div></section>
<section id="feedback"><h2>留下審查意見</h2><p>沒有排練可以先留白。填完下載 JSON 再交回聊天即可。內容只留在本頁，沒有上傳功能；關閉前請下載。</p><form id="reviewForm"><div class="formgrid"><label>喜歡的風格<select id="style"><option value="undecided">尚未決定</option><option>白底學術</option><option>深色科技</option><option>暖色雜誌</option></select></label><label>目前看法<select id="verdict"><option value="not_reviewed">尚未審查完</option><option value="keep">可保留</option><option value="revise">修改後再看</option></select></label><label>自己實際講了幾分鐘（可留白）<input id="minutes" type="number" min="0" max="120" step="0.1" placeholder="例如 15.2"></label><label>最卡的頁碼（可留白）<input id="pages" type="text" placeholder="例如 12、13"></label></div><p><label>哪一句不好懂，或想優先修改什麼？<textarea id="comments" placeholder="也可以先告訴我喜歡哪一款風格。"></textarea></label></p><button type="submit">下載審查意見</button><p id="status" role="status"></p></form><p><a href="審查意見.md">也可填文字版意見表</a></p></section>
<footer>簡報審查版，不改 Kaggle 選定。模型研究停在 R106。文件、影片與程式的身分見同層 manifest.json。</footer></main><script id="reviewData" type="application/json">__META__</script><script>
const data=JSON.parse(document.getElementById('reviewData').textContent);document.getElementById('openAll').onclick=()=>document.querySelectorAll('.page').forEach(e=>e.open=true);document.getElementById('closeAll').onclick=()=>document.querySelectorAll('.page').forEach(e=>e.open=false);document.getElementById('reviewForm').addEventListener('submit',e=>{e.preventDefault();const value=id=>document.getElementById(id).value;const out={...data,reviewed_at:new Date().toISOString(),preferred_style:value('style'),verdict:value('verdict'),human_rehearsal_minutes:value('minutes')===''?null:Number(value('minutes')),difficult_pages:value('pages'),comments:value('comments')};const blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json;charset=utf-8'});const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='v22_審查意見.json';a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);document.getElementById('status').textContent='已產生下載檔，請保存後交回聊天。';});
</script></body></html>'''
template=template.replace('__STYLES__',''.join(styles)).replace('__NOTES__',''.join(notes)).replace('__CHANGES__',''.join(changes)).replace('__META__',json.dumps(metadata,ensure_ascii=False).replace('<','\\u003c'))
write('index.html',template)
write('本次修改.md','# 本次審查版修改\n\n'+ '\n\n'.join(f"## 第{x['page']}頁\n\n{x['reason']}" for x in c['changes'])+'\n\n主講4,065漢字，英文名稱由17次減至12次。規劃14:45並非真人計時。原簡報、模型、價格與正式報告保留。影片與審查檔是本輪新增交付，尚未更新至GitHub。\n')
write('04_程式與證據/複製來源.json',json.dumps(copies,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'review_folder':str(OUT),'copied_files':len(copies)},ensure_ascii=False))
