from pathlib import Path
import json, hashlib, re
HERE=Path(__file__).resolve().parent
p=HERE/'content/proposed_scripts.json'
assert hashlib.sha256(p.read_bytes()).hexdigest()=='0308d933d25f27197badef718413c47b4e235d17516cfa163e8903fae378161d'
d=json.loads(p.read_text(encoding='utf-8'))
remove='Public的精確分數是1,998,568.5706元，這裡以約兩百萬元說明意義。'
assert d['slides'][16]['script'].count(remove)==1
d['slides'][16]['script']=d['slides'][16]['script'].replace(remove,'').strip()
for s in d['slides']:s['script']=s['script'].replace('；','。')
count=sum(len(re.findall(r'[\u4e00-\u9fff]',s['script'])) for s in d['slides'])
assert count==4051
with (HERE/'content/approved_scripts.json').open('x',encoding='utf-8') as f:json.dump(d,f,ensure_ascii=False,indent=2)
review={'reviewer':'root','scope':'Read all 20 proposed scripts against the preserved v22 methodology; author agent proposal plus root review, not a separate independent review.', 'edits':['Remove redundant raw precision from slide 17 spoken text; exact Public remains on slide.','Replace semicolons with sentence boundaries.'], 'approved_main_han':count,'previous_spoken_main_han':4981,'reduction_han':930,'preserved':['Model and preprocessing rationale','Three branches, R then S, final median','Temporal fit/query separation and grouped OOF distinction','Unknown Private, reused-window selection bias and reproduction limits'], 'human_rehearsal_measured':False}
with (HERE/'content/editor_review.json').open('x',encoding='utf-8') as f:json.dump(review,f,ensure_ascii=False,indent=2)
print(count)
